import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from flask import Flask, request
import numpy


class Aggregator(Flask):
    
    def _clear(self):
        self.state = {}
        self.data = {}
        
    def __init__(self, eval_host, eval_port, level=logging.INFO):
        super(Aggregator, self).__init__("Aggregator")
        self.logger.setLevel(level)
        self.eval_host = eval_host
        self.eval_port = eval_port
        self._clear()

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
        
        @self.route("/flush", methods=["POST"])
        def flush():
            self.logger.info("Flushing tables")
            for model_name, epoch in self.state.items():
                self._send_model(model_name)
            self._clear()
            return "OK"
        
        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":
                return "Aggregator server"
            elif request.method == "POST":
                j = request.get_json()
                model_name = j["metadata"]["model_name"]
                op_name = j["metadata"]["op_name"]
                mode = j["metadata"]["mode"]
                values = j["values"]
                if model_name in self.state and self.state[model_name]["epoch"] != j["metadata"]["epoch"]:
                    self._send_model(j["metadata"]["model_name"])
                self.state[model_name] = j["metadata"]
                self.data[model_name] = self.data.get(model_name, {})
                self.data[model_name][op_name] = self.data[model_name].get(op_name, {})
                self.data[model_name][op_name][mode] = self.data[model_name][op_name].get(mode, {})
                for group_name, slots in j["values"].items():
                    self.data[model_name][op_name][mode][group_name] = self.data[model_name][op_name][mode].get(group_name, {})
                    for slot_name, value in slots.items():
                        self.data[model_name][op_name][mode][group_name][slot_name] = self.data[model_name][op_name][mode][group_name].get(slot_name, [])
                        self.data[model_name][op_name][mode][group_name][slot_name].append(value)
            return "OK"
        
    def _send_model(self, model_name):
        self.logger.info("Sending data for model %s", model_name)
        for op_name, modes in self.data[model_name].items():
            for mode, groups in modes.items():
                for group_name, slots in groups.items():
                    for slot_name, vals in slots.items():
                        metadata = {k : v for k, v in self.state[model_name].items()}
                        metadata["op_name"] = op_name
                        metadata["mode"] = mode
                        metadata["group_name"] = group_name
                        metadata["slot_name"] = slot_name
                        metadata["epoch"] = self.state[model_name]["epoch"]
                        self.logger.info("Sending epoch '%(epoch)d' of mode '%(mode)s' for model/operation/group/slot '%(model_name)s/%(op_name)s/%(group_name)s/%(slot_name)s'", metadata)
                        vals = [numpy.asarray(d) for d in vals]
                        data = numpy.concatenate(vals)
                        r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps({"data" : data.tolist(), "metadata" : metadata}).encode()) #self.state[model_name]}).encode())
                        urlopen(r)
        self.data[model_name] = {}


def create_server(evaluator_host, evaluator_port):
    return Aggregator(evaluator_host, evaluator_port, logging.INFO)
