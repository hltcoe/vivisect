from urllib.request import urlopen, Request
import json
from functools import partial
import logging
from .vivisect_types import ModelMetadata, OperationMetadata, ArrayMetadata
from . import pytorch as ppy
from . import tensorflow as tpy
from . import mxnet as mpy


from uuid import uuid4


# get_ops(model) -> list[ops]
# attach(op, cb)
# get_params(op) -> list[arrays]
def flush(host, port):
    r = Request("http://{}:{}/flush".format(host, port), method="POST")
    urlopen(r)

    
def clear(host, port):
    r = Request("http://{}:{}/clear".format(host, port), method="POST")
    urlopen(r)    

#register_classification_task(name="mlp_train", targets=y_train, model_pattern="PyTorch MLP Model", layer_pattern=".*")    
def register_classification_targets(host, port, name, targets, model_pattern, layer_pattern):
    j = {"values" : targets.tolist(), "name" : name, "model_pattern" : model_pattern, "layer_pattern" : layer_pattern}
    r = Request("http://{}:{}/register_classification_targets".format(host, port),
                headers={"Content-Type" : "application/json"},
                data=json.dumps(j).encode(),
                method="POST")
    urlopen(r)


def register_clustering_targets(host, port, name, targets, model_pattern, layer_pattern):
    j = {"values" : targets.tolist(), "name" : name, "model_pattern" : model_pattern, "layer_pattern" : layer_pattern}
    r = Request("http://{}:{}/register_clustering_targets".format(host, port),
                headers={"Content-Type" : "application/json"},
                data=json.dumps(j).encode(),
                method="POST")
    urlopen(r)

    
def _forward_callback(operation, inputs, outputs, when, model, host, port):
    if when(model, operation):
        metadata = {k : v for k, v in list(model._vivisect.items()) + list(operation._vivisect.items())}
        metadata["mode"] = "forward"
        r = Request("http://{}:{}".format(host, port),
                    method="POST",
                    headers={"Content-Type" : "application/json"},
                    data=json.dumps({"values" : {"inputs" : inputs, "outputs" : outputs},
                                     "metadata" : metadata,
                    }).encode())
        urlopen(r)


def _backward_callback(operation, grad_input, grad_output, when, model, host, port):
    if when(model, operation):
        metadata = {k : v for k, v in list(model._vivisect.items()) + list(operation._vivisect.items())}
        metadata["mode"] = "backward"
        r = Request("http://{}:{}".format(host, port),
                    method="POST",
                    headers={"Content-Type" : "application/json"},
                    data=json.dumps({"values" : {"grad_input" : grad_input, "grad_output" : grad_output},
                                     "metadata" : metadata,
                    }).encode())
        urlopen(r)


def _parameter_callback(operation, parameters, when, model, host, port):
    if when(model, operation):
        metadata = {k : v for k, v in list(model._vivisect.items()) + list(operation._vivisect.items())}
        metadata["mode"] = "parameters"
        r = Request("http://{}:{}".format(host, port),
                    method="POST",
                    headers={"Content-Type" : "application/json"},
                    data=json.dumps({"values" : {"parameters" : parameters},
                                     "metadata" : metadata,
                    }).encode())
        urlopen(r)


def get_model_info(model):
    for fw in [ppy, mpy, tpy]:
        if isinstance(model, fw.model_types):
            ops = fw.get_operation_names(model)
            params = fw.get_parameter_names(model)
            return (ops, params)
    raise Exception("No way to treat '{}' as a model".format(type(model)))


def probe(model, host, port, which=lambda m, o : True, when=lambda m, o, a, b : True, **argdict):
    
    forward_callback = partial(_forward_callback, when=when, model=model, host=host, port=port)
    backward_callback = partial(_backward_callback, when=when, model=model, host=host, port=port)
    parameter_callback = partial(_parameter_callback, when=when, model=model, host=host, port=port)
    
    model._vivisect = getattr(model, "_vivisect", {})
    for k, v in argdict.items():
        assert(isinstance(v, (int, float, str)))
        model._vivisect[k] = v
        
    model._vivisect["model_name"] = model._vivisect.get("model_name", str(uuid4()))
    model._vivisect["op_name"] = model._vivisect["model_name"]
    
    for fw in [ppy, mpy, tpy]:
        if isinstance(model, fw.model_types):
            fw.parameter_attach(model, parameter_callback)
            for name, operation in fw.get_ops(model):
                operation._vivisect = getattr(operation, "_vivisect", {})
                operation._vivisect["op_name"] = name
                if which(model, operation):
                    logging.info("Monitoring operation '%s'", name)
                    fw.forward_attach(operation, forward_callback)
                    fw.backward_attach(operation, backward_callback)
                else:
                    logging.info("Not monitoring operation '%s'", name)
            return True
    raise Exception("Vivisect doesn't know how to handle a model of type '{}'".format(type(model)))


def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs):
    model._vivisect["epoch"] = 0
    for fw in [ppy, mpy, tpy]:
        if isinstance(model, fw.model_types):
            fw.train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs)
            return True
    raise Exception("Vivisect doesn't know how to handle a model of type '{}'".format(type(model)))
