import sqlite3
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
import numpy
from flask import Flask, request
import functools


class Evaluator(Flask):


    def _clear(self):
        self._targets = {}
        

    
    def __init__(self, frontend_host, frontend_port, level=logging.INFO):
        super(Evaluator, self).__init__("Evaluator")
        self.logger.setLevel(level)
        self._metrics = {}
        self._targets = {}
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port

        @self.route("/clear", methods=["POST"])
        def clear():
            self.logger.info("Clearing system")
            self._clear()
            r = Request("http://{}:{}/clear".format(self.frontend_host, self.frontend_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        )
            urlopen(r)            
            return "OK"


        
        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":                
                return "Evaluator server"
            elif request.method == "POST":                
                j = request.get_json()
                self.logger.info("Received epoch %(epoch)s of model '%(model_name)s', operation '%(op_name)s', group '%(group_name)s', slot '%(slot_name)s'", j["metadata"])
                for metric_name, callback in self._metrics.items():
                    self.logger.info("Calculating %s", metric_name)
                    retval = {"metadata" : {k : v for k, v in j["metadata"].items()}}
                    retval["metric_name"] = metric_name
                    retval["metric_value"] = callback(j["data"], j["metadata"])
                    if retval["metric_value"] != None:
                        r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps(retval).encode())
                        urlopen(r)
                return "OK"

    def register_metric(self, name, callback):        
        self._metrics[name] = callback

    def register_targets(self, name, vals):
        self._targets[name] = vals


def average_absolute_activation(inputs, outputs, metadata):
    total = 0.0
    count = 0
    for batches in outputs:
        for batch in batches:
            np = numpy.abs(numpy.array(batch))
            total += np.sum()
            count += functools.reduce(lambda x, y : x * y, np.shape)
    return 0.0 if count == 0 else total / count


def mean(data, metadata):    
    return numpy.asarray(data).flatten().mean()


def standard_deviation(data, metadata):
    return numpy.std(numpy.asarray(data).flatten(), axis=0)


def classify(data, metadata):
    if "dense" in metadata["op_name"] and metadata["slot_name"] == "output":
        data = numpy.asarray(data)
        axis = 1
        
        print(data.shape)
        return 1.0


def create_server(frontend_host, frontend_port):
    server = Evaluator(frontend_host, frontend_port, logging.INFO)
    server.register_metric("Mean", mean)
    server.register_metric("Standard deviation", standard_deviation)
    server.register_metric("Classify", classify)
    return server
