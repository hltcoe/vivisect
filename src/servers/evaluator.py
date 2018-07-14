import sqlite3
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
import numpy
from scipy.stats import multinomial
from flask import Flask, request
import functools
from sklearn import linear_model, metrics, cluster
import re

class Evaluator(Flask):

    def _classify(self, data, targets):
        if not re.match(targets["layer_pattern"], data["metadata"]["group_name"]):
            return None
        d = numpy.asarray(data["data"])
        t = numpy.asarray(targets["values"])
        if d.shape[0] == t.shape[0]:
            d = numpy.asarray([x.flatten() for x in d])
            self.logger.info("%s %s", d.shape, t.shape)
            
            classifier = linear_model.SGDClassifier(max_iter=1000)
            classifier.fit(d, t)
            p = classifier.predict(d)
            return metrics.f1_score(t, p.tolist(), average="macro")

    def _cluster(self, data, targets):
        if not re.match(targets["layer_pattern"], data["metadata"]["group_name"]):
            return None
        d = numpy.asarray(data["data"])
        t = numpy.asarray(targets["values"])
        if d.shape[0] == t.shape[0]:
            d = numpy.asarray([x.flatten() for x in d])
            clusterer = cluster.KMeans(n_clusters=len(set(targets["values"]))).fit(d)
            return metrics.mutual_info_score(t, clusterer.labels_)
        
    def _clear(self):
        self._clustering_targets = {}
        self._classification_targets = {}        
            
    def __init__(self, frontend_host, frontend_port, level=logging.INFO):
        super(Evaluator, self).__init__("Evaluator")
        self.logger.setLevel(level)
        self._intrinsic_metrics = {}
        self._clustering_targets = {}
        self._classification_targets = {}        
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
        
        @self.route("/register_clustering_targets", methods=["POST"])
        def register_clustering_targets():
            j = request.get_json()
            self.logger.info("Processing register call for clustering '%(name)s'", j)
            self._clustering_targets[j["name"]] = j
            return "OK"

        @self.route("/register_classification_targets", methods=["POST"])
        def register_classification_targets():
            j = request.get_json()
            self.logger.info("Processing register call for classification '%(name)s'", j)
            self._classification_targets[j["name"]] = j
            return "OK"
        
        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":                
                return "Evaluator server"
            elif request.method == "POST":                
                j = request.get_json()
                self.logger.info("Received epoch %(epoch)s of model '%(model_name)s', operation '%(op_name)s', group '%(group_name)s', slot '%(slot_name)s'", j["metadata"])
                for metric_name, callback in self._intrinsic_metrics.items():
                    self.logger.info("Calculating %s", metric_name)
                    retval = {"metadata" : {k : v for k, v in j["metadata"].items()}}
                    retval["metric_name"] = "Intrinsic: {}".format(metric_name)
                    retval["metric_value"] = callback(j["data"], j["metadata"])
                    if retval["metric_value"] != None:
                        r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps(retval).encode())
                        urlopen(r)
                for metric_name, spec in self._classification_targets.items():
                    self.logger.info("Calculating %s", metric_name)
                    d = numpy.asarray(spec["values"])
                    self.logger.info("%s", d.shape)
                    retval = {"metadata" : {k : v for k, v in j["metadata"].items()}}
                    retval["metric_name"] = "Classification: {}".format(metric_name)
                    retval["metric_value"] = self._classify(j, spec)
                    
                    if retval["metric_value"] != None:
                        r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps(retval).encode())
                        urlopen(r)
                for metric_name, spec in self._clustering_targets.items():
                    self.logger.info("Calculating %s", metric_name)
                    retval = {"metadata" : {k : v for k, v in j["metadata"].items()}}
                    retval["metric_name"] = "Clustering: {}".format(metric_name)
                    retval["metric_value"] = self._cluster(j, spec)
                    if retval["metric_value"] != None:
                        r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps(retval).encode())
                        urlopen(r)


                return "OK"

    def _register_intrinsic_metric(self, name, callback):        
        self._intrinsic_metrics[name] = callback

    # def _register_classification_targets(self, name, spec):
    #     self._classification_targets[name] = spec

    # def _register_clustering_targets(self, name, spec):
    #     self._clustering_targets[name] = spec


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


def entropy(data, metadata):    
    np = numpy.asarray(data).flatten()
    m = multinomial(n=1, p=np / np.sum())
    e = m.entropy()
    r = e if isinstance(e, float) else e.tolist()    
    return (r if isinstance(r, float) else 0.0)


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
    server._register_intrinsic_metric("Mean", mean)
    server._register_intrinsic_metric("Standard deviation", standard_deviation)
    #server._register_intrinsic_metric("Entropy", entropy)
    return server
