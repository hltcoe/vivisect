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
        #self.state = {"forward" : {},
        #              "backward" : {},
        #              "parameters" : {},
        #}
        #self.data = {"forward" : {},
        #             "backward" : {},
        #             "parameters" : {},                         
        #}
            
        # @self.route("/clear", methods=["POST"])
        # def clear():
        #     self.state = {"forward" : {},
        #                   "backward" : {},
        #                   "parameters" : {},
        #     }
        #     self.data = {"forward" : {},
        #                  "backward" : {},
        #                  "parameters" : {},                         
        #     }
        #     return "OK"


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
                # for op_name, slots in self.data[model_name].items():
                   #     for slot_name, vals in slots.items():
                   #         self.state[model_name]["op_name"] = op_name
                   #         self.state[model_name]["slot_name"] = slot_name
                   #         epoch = self.state[model_name]["epoch"]
                   #         vals = [numpy.asarray(d) for d in vals]
                   #         self.logger.info("Sending epoch %d of operation/slot '%s/%s' with shape %s", epoch, op_name, slot_name, vals[0].shape)
                   #         data = numpy.concatenate(vals)
                   #         r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                   #                     method="POST",
                   #                     headers={"Content-Type" : "application/json"},
                   #                     data=json.dumps({"data" : data.tolist(), "metadata" : self.state[model_name]}).encode())
                   #         urlopen(r)
                #self.state = {"forward" : {},
                #              "backward" : {},
                #              "parameters" : {},
                #}
                #self.data = {"forward" : {},
                #             "backward" : {},
                #             "parameters" : {},                         
                #}
            self._clear()
            return "OK"


        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":
                return "Aggregator server"
            elif request.method == "POST":
                j = request.get_json()
                model_name = j["metadata"]["model_name"]
                #self.logger.info(model_name)
                
                op_name = j["metadata"]["op_name"]
                #slot_name = j["metadata"]["slot_name"]
                mode = j["metadata"]["mode"]
                values = j["values"]

                #"forward" if "outputs" in j else "backward" if "grad_outputs" in j else "parameters"
                
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
                # if mode == "forward":
                #     # The packet represents activations (and the first dimensions correspond to batches)
                #     if model_name not in self.state:
                #         self.logger.info("Adding entry for new model '%s'", model_name)
                #         self.state[model_name] = j["metadata"]
                #         self.data[model_name] = {}                    
                #     elif self.state[model_name]["epoch"] != epoch:
                #         self.logger.info("Model '%s' seems to have finished epoch %d", model_name, self.state[model_name]["epoch"])
                #         for op_name, slots in self.data[model_name].items():
                #             for slot_name, vals in slots.items():                                                        
                #                 self.state[model_name]["op_name"] = op_name
                #                 epoch = self.state[model_name]["epoch"]
                #                 self.state[model_name]["slot_name"] = slot_name
                #                 vals = [numpy.asarray(d) for d in vals]
                #                 self.logger.info("Sending epoch %d of operation/slot '%s/%s' with shape %s", epoch, op_name, slot_name, vals[0].shape)
                #                 data = numpy.concatenate(vals)
                #                 r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                #                             method="POST",
                #                             headers={"Content-Type" : "application/json"},
                #                             data=json.dumps({"data" : data.tolist(), "metadata" : self.state[model_name]}).encode())
                #                 urlopen(r)
                #         self.state[model_name] = j["metadata"]
                #         self.data[model_name] = {}
                #     self.data[model_name] = self.data.get(model_name, {})                
                #     self.data[model_name][op_name] = self.data[model_name].get(op_name, {})
                #     op_name = j["metadata"]["op_name"]
                #     self.data[model_name][op_name] = self.data[model_name].get(op_name, {})
                #     for k, d in j["outputs"].items():
                #         self.data[model_name][op_name][k] = self.data[model_name][op_name].get(k, [])
                #         self.data[model_name][op_name][k].append(d)
                # elif mode == "backward":
                #     # The packet represents gradients (and the first dimensions correspond to batches)
                #     if model_name not in self.state:
                #         self.logger.info("Adding entry for new model '%s'", model_name)
                #         self.state[model_name] = j["metadata"]
                #         self.data[model_name] = {}                    
                #     elif self.state[model_name]["epoch"] != epoch:
                #         self.logger.info("Model '%s' seems to have finished epoch %d", model_name, self.state[model_name]["epoch"])
                #         for op_name, slots in self.data[model_name].items():
                #             for slot_name, vals in slots.items():                                                        
                #                 self.state[model_name]["op_name"] = op_name
                #                 epoch = self.state[model_name]["epoch"]
                #                 self.state[model_name]["slot_name"] = slot_name
                #                 vals = [numpy.asarray(d) for d in vals]
                #                 self.logger.info("Sending epoch %d of operation/slot '%s/%s' with shape %s", epoch, op_name, slot_name, vals[0].shape)
                #                 data = numpy.concatenate(vals)
                #                 r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                #                             method="POST",
                #                             headers={"Content-Type" : "application/json"},
                #                             data=json.dumps({"data" : data.tolist(), "metadata" : self.state[model_name]}).encode())
                #                 urlopen(r)
                #         self.state[model_name] = j["metadata"]
                #         self.data[model_name] = {}
                #     self.data[model_name] = self.data.get(model_name, {})                
                #     self.data[model_name][op_name] = self.data[model_name].get(op_name, {})
                #     op_name = j["metadata"]["op_name"]
                #     self.data[model_name][op_name] = self.data[model_name].get(op_name, {})
                #     for k, d in j["grad_outputs"].items():
                #         self.data[model_name][op_name][k] = self.data[model_name][op_name].get(k, [])
                #         self.data[model_name][op_name][k].append(d)
                # elif mode == "parameters":
                #     # The packet represents parameter-states (which have no batch-dimensions)
                #     pass
                # else:
                #     # Otherwise, just pass along the request as-is: it's probably intended for a downstream server
                #     r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                #                 method="POST",
                #                 headers={"Content-Type" : "application/json"},
                #                 data=json.dumps(j).encode())
                #     urlopen(r)
                # return "OK"

    def _send_model(self, model_name):
        self.logger.info("Sending data for model %s", model_name)
        for op_name, modes in self.data[model_name].items():
            for mode, groups in modes.items():
                for group_name, slots in groups.items():
                    #self.logger.info("%s", slots)
                    #continue
                    for slot_name, vals in slots.items():
                        metadata = {k : v for k, v in self.state[model_name].items()}
                        metadata["op_name"] = op_name
                        metadata["mode"] = mode
                        metadata["group_name"] = group_name
                        metadata["slot_name"] = slot_name
                        metadata["epoch"] = self.state[model_name]["epoch"]
                        self.logger.info("Sending epoch '%(epoch)d' of mode '%(mode)s' for model/operation/group/slot '%(model_name)s/%(op_name)s/%(group_name)s/%(slot_name)s'", metadata)
                        #self.state[model_name]["op_name"] = op_name
                        #self.state[model_name]["slot_name"] = slot_name
                        #epoch = self.state[model_name]["epoch"]
                        vals = [numpy.asarray(d) for d in vals]
                        #self.logger.info("%s", vals)
                        data = numpy.concatenate(vals)
                        #metadata["shape"] = data.shape

                        #self.logger.info("Sending epoch %(epoch)d of mode %(mode)s for model/operation/slot '%(model_name)s/%(op_name)s/%(slot_name)s' with shape %(shape)s", metadata)

                        r = Request("http://{}:{}".format(self.eval_host, self.eval_port),
                                    method="POST",
                                    headers={"Content-Type" : "application/json"},
                                    data=json.dumps({"data" : data.tolist(), "metadata" : metadata}).encode()) #self.state[model_name]}).encode())
                        urlopen(r)
        self.data[model_name] = {}
                
def create_server(eval_host, eval_port):
    return Aggregator(eval_host, eval_port, logging.INFO)
