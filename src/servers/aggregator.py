import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from flask import Flask, request


class Aggregator(Flask):
    
    def __init__(self, eval_host, eval_port):
        super(Aggregator, self).__init__("Aggregator")
        self.eval_host = eval_host
        self.eval_port = eval_port
        self.data = {}
        self.state = {}

        @self.route("/clear", methods=["POST"])
        def clear():
            self.state = {}
            self.data = {}
            return "OK"
            
        @self.route("/flush", methods=["POST"])
        def flush():
            logging.info("Flushing tables")
            for model_id, iteration in self.state.items():
                for op_name, vals in self.data[model_id].items():
                    self.state[model_id]["op_name"] = op_name                    
                    r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                                method="POST",
                                headers={"Content-Type" : "application/json"},
                                data=json.dumps({"data" : vals, "metadata" : self.state[model_id]}).encode())
                    urlopen(r)
            self.state = {}
            self.data = {}
            return "OK"
        
        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":
                return "Aggregator server"
            elif request.method == "POST":
                j = request.get_json()

                model_name = j["metadata"]["model_name"]
                op_name = j["metadata"]["op_name"]
                iteration = max(1, j["metadata"]["iteration"])
                j["metadata"]["iteration"] = iteration
                logging.info("Model: %s, Iteration: %s", model_name, iteration)
                print("Model: %s, Iteration: %s" % (model_name, iteration))
                if model_name not in self.state:
                    self.state[model_name] = j["metadata"]
                    self.data[model_name] = {}
                elif self.state[model_name]["iteration"] != iteration:
                    for op_name, vals in self.data[model_name].items():
                        self.state[model_name]["op_name"] = op_name
                        r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps({"data" : vals, "metadata" : self.state[model_name]}).encode())
                        urlopen(r)
                    self.state[model_name] = j["metadata"]
                    self.data[model_name] = {}
                self.data[model_name] = self.data.get(model_name, {})
                self.data[model_name][op_name] = self.data[model_name].get(op_name, [])
                self.data[model_name][op_name].append((j["inputs"], j["outputs"]))
                return "OK"
                
def create_server(eval_host, eval_port):
    return Aggregator(eval_host, eval_port)
