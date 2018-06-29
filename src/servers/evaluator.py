import sqlite3
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
import numpy
from flask import Flask, request
import functools


class Evaluator(Flask):
    def __init__(self, frontend_host, frontend_port):
        super(Evaluator, self).__init__("Evaluator")
        self._metrics = {}
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":                
                return "Evaluator server"
            elif request.method == "POST":                
                j = request.get_json()
                inputs = numpy.asarray([x[0] for x in j["data"]])
                outputs = [x[1] for x in j["data"]]
                for name, callback in self._metrics.items():
                    retval = {"metadata" : {k : v for k, v in j["metadata"].items()}}
                    retval["metric_name"] = name
                    retval["metric_value"] = callback(inputs, outputs, j["metadata"])
                    r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                method="POST",
                                headers={"Content-Type" : "application/json"},
                                data=json.dumps(retval).encode())
                    urlopen(r)
                return "OK"

    def register_metric(self, name, callback):
        self._metrics[name] = callback


def average_activation(inputs, outputs, metadata):
    total = 0.0
    count = 0
    for batches in outputs:
        for batch in batches:
            np = numpy.array(batch)
            total += np.sum()
            count += functools.reduce(lambda x, y : x * y, np.shape)
    return 0.0 if count == 0 else total / count


def create_server(frontend_host, frontend_port):
    server = Evaluator(frontend_host, frontend_port)
    server.register_metric("Average activation", average_activation)
    return server
