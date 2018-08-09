import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from flask import Flask, request
import numpy
import jinja2


status_template = jinja2.Template("""
<html>
  <head>
<script type="text/javascript">
function load()
{
setTimeout("window.open(self.location, '_self');", 5000);
}
</script>
  </head>
  <body onload="load()">
    <table>
    <tr><td>Trigger keys: </td>
    {% for key in server.trigger_fields %}
      <td>{{ key }}</td>
    {% endfor %}
    </tr>
    <tr><td>Preserved keys: </td>
    {% for key in server.cdp_fields %}
      <td>{{ key }}</td>
    {% endfor %}
    </tr>
    {% for name, val in server.state.items() %}
      <tr><td>{{ name }}</td><td>{{ val }}</td></tr>
    {% endfor %}
    {% for name in server.data.keys() %}
      <tr><td>{{ name }}</td></tr>
    {% endfor %}
    </table>
  </body>
</html>
""")


class Aggregator(Flask):
    
    def _clear(self):
        self.state = {}
        self.data = {}
        
    def __init__(self, eval_host, eval_port, trigger_fields=["epoch"], cdp_fields=["epoch", "model_name", "array_name", "array_type", "operation_name", "batch_axis"], level=logging.INFO):
        super(Aggregator, self).__init__("Aggregator")
        self.logger.setLevel(level)
        self.eval_host = eval_host
        self.eval_port = eval_port
        self.trigger_fields = trigger_fields
        self.cdp_fields = cdp_fields
        self.data = {}
        self.state = {}
        self._clear()

        @self.route("/status", methods=["GET"])
        def status():
            return status_template.render(server=self)
        
        @self.route("/register_classification_targets", methods=["POST"])
        def register_classification_targets():
            j = request.get_json()
            self.logger.info("Passing along a registration call for '%(name)s'", j)
            r = Request("http://{}:{}/register_classification_targets".format(self.eval_host, self.eval_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        data=json.dumps(j).encode(),
                        )
            urlopen(r)            
            return "OK"

        @self.route("/register_clustering_targets", methods=["POST"])
        def register_clustering_targets():
            j = request.get_json()
            self.logger.info("Passing along a registration call for clustering '%(name)s'", j)
            r = Request("http://{}:{}/register_clustering_targets".format(self.eval_host, self.eval_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        data=json.dumps(j).encode(),
                        )
            urlopen(r)            
            return "OK"

        @self.route("/set_trigger_fields", methods=["POST"])
        def set_trigger_fields():
            self.logger.info("Setting trigger fields")
            j = request.get_json()
            assert(isinstance(j, list) and all([isinstance(x, str) for x in j]))
            self.trigger_fields = j + ["batch_axis"]
            return "OK"
        
        @self.route("/clear", methods=["POST"])
        def clear():
            self.logger.info("Clearing system")
            self._clear()
            r = Request("http://{}:{}/clear".format(self.eval_host, self.eval_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        )
            urlopen(r)            
            return "OK"

        @self.route("/remove", methods=["POST"])
        def remove():
            j = request.get_json()
            self.logger.info("Removing information for model '%s'", j["model_name"])
            try:
                del self.data[j["model_name"]]
            except:
                pass
            try:
                del self.state[j["model_name"]]
            except:
                pass
            r = Request("http://{}:{}/remove".format(self.eval_host, self.eval_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        data=json.dumps(j).encode())
            urlopen(r)            
            return "OK"
        
        @self.route("/flush", methods=["POST"])
        def flush():
            self.logger.info("Flushing tables")
            for model_name, _ in self.state.items():
                self._send_model(model_name)
            self._clear()
            return "OK"

        @self.route("/activation_input", methods=["POST"])
        def activation_input():
            j = request.get_json()
            self.logger.info("Processing %s", j["metadata"])
            self._process(j["array"], j["metadata"], "activation_input")
            return "OK"

        @self.route("/activation_output", methods=["POST"])
        def activation_output():
            j = request.get_json()
            self.logger.info("Processing %s", j["metadata"])
            #self.logger.info("Processing %s", j["array"])
            self._process(j["array"], j["metadata"], "activation_output")
            return "OK"

        @self.route("/gradient_input", methods=["POST"])
        def gradient_input():
            j = request.get_json()
            self.logger.info("Processing %s", j["metadata"])
            self._process(j["array"], j["metadata"], "gradient_input")
            return "OK"

        @self.route("/gradient_output", methods=["POST"])
        def gradient_output():
            j = request.get_json()
            self.logger.info("Processing %s", j["metadata"])
            self._process(j["array"], j["metadata"], "gradient_output")
            return "OK"

        @self.route("/parameter", methods=["POST"])
        def parameter():
            j = request.get_json()
            self.logger.info("Processing %s", j["metadata"])
            self._process(j["array"], j["metadata"], "parameter")
            return "OK"
        
        @self.route("/", methods=["GET"])
        def handle():
            return "Aggregator server"

    def _process(self, array, metadata, array_type):
        trigger_key = {metadata.get(k, None) for k in self.trigger_fields}
        cdp_key = tuple([metadata.get(k, None) for k in self.cdp_fields])
        model_name = metadata["model_name"]
        if self.state.get(model_name, trigger_key) != trigger_key:
            self._send_model(model_name)
            del self.state[model_name]
            self.data[model_name] = {}
        self.state[model_name] = trigger_key
        self.data[model_name] = self.data.get(model_name, {})
        self.data[model_name][cdp_key] = self.data[model_name].get(cdp_key, [])
        self.data[model_name][cdp_key].append(array)
        
    def _send_model(self, model_name):
        self.logger.info("Sending aggregated CDPs for model %s", model_name)
        for cdp_key, arrays in self.data[model_name].items():
            j = {"data" : arrays, "metadata" : {k : v for k, v in zip(self.cdp_fields, cdp_key)}}
            self.logger.info("Creating CDP for %s", j["metadata"])
            r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        data=json.dumps(j).encode())
            urlopen(r)
        return None


def create_server(evaluator_host, evaluator_port):
    return Aggregator(evaluator_host, evaluator_port, level=logging.INFO)
