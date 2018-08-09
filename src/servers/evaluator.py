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
        classifier = linear_model.SGDClassifier(max_iter=100)
        classifier.fit(data, targets)
        p = classifier.predict(data)
        return metrics.f1_score(targets, p.tolist(), average="macro")

    def _cluster(self, data, targets):
        clusterer = cluster.KMeans(n_clusters=len(set(targets.tolist()))).fit(data)
        return metrics.mutual_info_score(targets, clusterer.labels_)
        
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

        @self.route("/remove", methods=["POST"])
        def remove():
            j = request.get_json()
            self.logger.info("Removing '%s'", j["model_name"])
            r = Request("http://{}:{}/remove".format(self.frontend_host, self.frontend_port),
                        method="POST",
                        headers={"Content-Type" : "application/json"},
                        data=json.dumps(j).encode())
            urlopen(r)            
            return "OK"
        
        @self.route("/register_clustering_targets", methods=["POST"])
        def register_clustering_targets():
            j = request.get_json()
            if isinstance(j["values"][0], list):
                assert(isinstance(j["values"][0][0], (int, float)))
                lens = [len(x) for x in j["values"]]
                max_len = max(lens)
                data = numpy.zeros(shape=(len(j["values"]), max_len))
                for i, (l, d) in enumerate(zip(lens, j["values"])):
                    data[i, 0:l] = d
                j["values"] = data #.squeeze()
            else:
                j["values"] = numpy.asarray(j["values"]) #.squeeze()
            self.logger.info("Processing register call for clustering '%s', shape %s", j["name"], j["values"].shape)
            self._clustering_targets[j["name"]] = j
            return "OK"

        @self.route("/register_classification_targets", methods=["POST"])
        def register_classification_targets():
            j = request.get_json()
            if isinstance(j["values"][0], list):
                assert(isinstance(j["values"][0][0], (int, float)))
                lens = [len(x) for x in j["values"]]
                max_len = max(lens)
                data = numpy.zeros(shape=(len(j["values"]), max_len))
                for i, (l, d) in enumerate(zip(lens, j["values"])):
                    data[i, 0:l] = d
                j["values"] = data #.squeeze()
            else:
                j["values"] = numpy.asarray(j["values"]) #.squeeze()
            self.logger.info("Processing register call for classification '%s', shape %s", j["name"], j["values"].shape)
            self._classification_targets[j["name"]] = j
            return "OK"
        
        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":                
                return "Evaluator server"
            elif request.method == "POST":                
                j = request.get_json()
                self.logger.info("%s", j["metadata"])
                #self.logger.info("Received CDP of array type '%(array_type)s' for epoch %(epoch)s of model '%(model_name)s', operation '%(operation_name)s', array '%(array_name)s'", j["metadata"])
                arrays = [numpy.asarray(x) for x in j["data"]]
                max_shapes = numpy.asarray([a.shape for a in arrays]).max(0)
                arrays = [numpy.pad(a, [(0, t - c) for c, t in zip(a.shape, max_shapes)], "constant") for a in arrays]
                ax = j["metadata"].get("batch_axis", 0) if len(arrays[0].shape) > 1 else 0
                ax = ax if isinstance(ax, int) else 0
                tarray = numpy.concatenate(arrays, axis=ax)
                array = tarray.swapaxes(0, ax)
                #print(array.shape, j["metadata"])
                # except:
                #     print(shapes)
                #     print([x.shape for x in arrays])
                #     print(j["metadata"])
                #     sys.exit()
                #     return "OK"
                
                for metric_name, callback in self._intrinsic_metrics.items():
                    self.logger.info("Calculating %s", metric_name)
                    retval = {k : v for k, v in j["metadata"].items()}
                    retval["metric_type"] = "intrinsic"
                    retval["metric_name"] = metric_name
                    retval["metric_value"] = callback(array, j["metadata"])
                    if retval["metric_value"] != None:
                        r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps({"metadata" : retval}).encode())
                        urlopen(r)
                for metric_name, spec in self._classification_targets.items():
                    print(array.shape, spec["values"].shape, metric_name, j["metadata"])
                    target_shape = spec["values"].shape
                    if array.shape[0:len(target_shape)] == target_shape:
                        
                        item_count = functools.reduce(lambda x, y : x * y, target_shape, 1)
                        #target_shape[0] if len(target_shape) == 1 else target_shape[0] * target_shape[1]
                        #functools.reduce(lambda x, y : x * y, target_shape, 1)
                        print(item_count)
                        feats = numpy.reshape(array, (item_count, -1))
                        targs = numpy.reshape(spec["values"], (item_count, -1)).flatten()
                        idx = [i for i, v in enumerate(targs) if v != 0]
                        feats = feats[idx]
                        targs = targs[idx]
                        print(feats.shape, targs.shape)
                        self.logger.info("Calculating %s", metric_name)
                        retval = {k : v for k, v in j["metadata"].items()}
                        retval["metric_type"] = "classification"
                        retval["metric_name"] = metric_name
                        retval["metric_value"] = self._classify(feats, targs)
                    
                        if retval["metric_value"] != None:
                            r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                                        method="POST",
                                        headers={"Content-Type" : "application/json"},
                                        data=json.dumps({"metadata" : retval}).encode())
                            urlopen(r)
                            
                # for metric_name, spec in self._clustering_targets.items():
                #     print(array.shape, spec["values"].shape[0], metric_name, j["metadata"])
                #     if array.shape[0] == spec["values"].shape[0]:                        
                #         self.logger.info("Calculating %s", metric_name)
                #         retval = {k : v for k, v in j["metadata"].items()}
                #         retval["metric_type"] = "clustering"
                #         retval["metric_name"] = metric_name
                #         retval["metric_value"] = self._cluster([r.flatten() for r in array], spec["values"])
                #         if retval["metric_value"] != None:
                #             r = Request("http://{}:{}".format(self.frontend_host, self.frontend_port),
                #                         method="POST",
                #                         headers={"Content-Type" : "application/json"},
                #                         data=json.dumps({"metadata" : retval}).encode())
                #             urlopen(r)
                return "OK"

    def _register_intrinsic_metric(self, name, callback):        
        self._intrinsic_metrics[name] = callback


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

def mean_abs(data, metadata):    
    return numpy.abs(numpy.asarray(data).flatten()).mean()


def standard_deviation(data, metadata):
    return numpy.std(numpy.asarray(data).flatten(), axis=0)


def create_server(frontend_host, frontend_port):
    server = Evaluator(frontend_host, frontend_port, logging.INFO)
    server._register_intrinsic_metric("Mean", mean)
    server._register_intrinsic_metric("Mean Absolute Value", mean_abs)
    server._register_intrinsic_metric("Standard deviation", standard_deviation)
    return server
