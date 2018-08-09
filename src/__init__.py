from urllib.request import urlopen, Request
import json
from functools import partial, wraps
import functools
import logging
from .vivisect_types import ModelInfo, OperationInfo, ArrayInfo
from . import pytorch as ppy
from . import gluon as gpy
#from . import mxnet as mpy
#from . import tensorflow as tpy

from uuid import uuid4

fws = [ppy, gpy]


default_which_operation = lambda model, operation : True
default_which_array = lambda model, operation, array_info : True
#default_when_op_filter = lambda model_info, operation_info : True
#default_when_array_filter = lambda model_info, operation_info, array_info : True
default_when = lambda model : True


def _post(host, port, url, json_data=None):
    r = Request("http://{}:{}/{}".format(host, port, url),
                headers={"Content-Type" : "application/json"},
                data=json.dumps(json_data).encode() if json_data else None,
                method="POST")
    urlopen(r)


flush = functools.partial(_post, url="flush")
clear = functools.partial(_post, url="clear")


def remove(host, port, model_name):
    r = Request("http://{}:{}/remove".format(host, port),
                headers={"Content-Type" : "application/json"},
                data=json.dumps({"model_name" : model_name}).encode(),
                method="POST")
    urlopen(r)

    
def register_classification_targets(host, port, name, targets, model_pattern, layer_pattern=".*"):
    j = {"values" : targets if isinstance(targets, list) else targets.tolist(), "name" : name, "model_pattern" : model_pattern, "layer_pattern" : layer_pattern}
    _post(host, port, "register_classification_targets", j)


def register_clustering_targets(host, port, name, targets, model_pattern, layer_pattern=".*"):
    j = {"values" : targets if isinstance(targets, list) else targets.tolist(), "name" : name, "model_pattern" : model_pattern, "layer_pattern" : layer_pattern}
    _post(host, port, "register_clustering_targets", j)


def _forward_callback(operation, inputs, outputs, when, which_array, model, host, port, batch_axis=0):
    if when(model, operation):
        metadata = {}
        for array_name, array in inputs.items():
            array_info = ArrayInfo(array_name=array_name, array_type="activation_input")
            if which_array(model, operation, array_info):
                j = {"metadata" : {"batch_axis" : batch_axis}, "array" : array}
                for x in [model._v, operation._v, array_info]:
                    for s in x.__slots__:
                        j["metadata"][s] = getattr(x, s)
                        
                _post(host, port, "activation_input", j)
        for array_name, array in outputs.items():
            array_info = ArrayInfo(array_name=array_name, array_type="activation_output")
            if which_array(model, operation, array_info):       
                j = {"metadata" : {"batch_axis" : batch_axis}, "array" : array}
                for x in [model._v, operation._v, array_info]:
                    for s in x.__slots__:
                        j["metadata"][s] = getattr(x, s)
                _post(host, port, "activation_output", j)                        


def _backward_callback(operation, grad_inputs, grad_outputs, when, which_array, model, host, port, batch_axis=0):
    if when(model, operation):        
        for array_name, array in grad_inputs.items():
            array_info = ArrayInfo(array_name=array_name, array_type="gradient_input")
            if which_array(model, operation, array_info):                
                j = {"metadata" : {"batch_axis" : batch_axis}, "array" : array}
                for x in [model._v, operation._v, array_info]:
                    for s in x.__slots__:
                        j["metadata"][s] = getattr(x, s)
                _post(host, port, "gradient_input", j)                        
            
        for array_name, array in grad_outputs.items():
            array_info = ArrayInfo(array_name=array_name, array_type="gradient_output")
            if which_array(model, operation, array_info):                
                j = {"metadata" : {"batch_axis" : batch_axis}, "array" : array}
                for x in [model._v, operation._v, array_info]:
                    for s in x.__slots__:
                        j["metadata"][s] = getattr(x, s)
                _post(host, port, "gradient_output", j)                        

            
def _parameter_callback(model, parameters, when, which_array, host, port):
    if True: #when(model):
        for array_name, array in parameters.items():
            array_info = ArrayInfo(array_name=array_name, array_type="parameter")
            if which_array(model, None, array_info):
                j = {"metadata" : {}, "array" : array}
                for x in [model._v, array_info]:
                    for s in x.__slots__:
                        j["metadata"][s] = getattr(x, s)
                _post(host, port, "parameter", j)                        

                
def probe(model_name, model, host, port, which=lambda m, o : True, when=default_when, which_array=default_which_array, parameters=True, forward=True, backward=True, batch_axis=0):
    forward_callback = partial(_forward_callback, when=when, which_array=which_array, model=model, host=host, port=port, batch_axis=batch_axis)
    backward_callback = partial(_backward_callback, when=when, which_array=which_array, model=model, host=host, port=port, batch_axis=batch_axis)
    parameter_callback = partial(_parameter_callback, when=when, which_array=which_array, host=host, port=port)
    model._v = ModelInfo(model_name=model_name)    
    for fw in fws:
        if isinstance(model, fw.model_types):
            if parameters:
                fw.parameter_attach(model, parameter_callback)
            for operation_name, operation in fw.get_ops(model):
                if not hasattr(operation, "_v"):
                    operation._v = OperationInfo(operation_name=operation_name, operation_type=str(type(operation)))
                if isinstance(operation._v, OperationInfo) and which(model, operation):
                    operation._v = OperationInfo(operation_name=operation_name, operation_type=str(type(operation)))
                    logging.info("Monitoring operation: %s", operation._v)
                    if forward:
                        fw.forward_attach(operation, forward_callback)
                    if backward:
                        fw.backward_attach(operation, backward_callback)
            return True
    raise Exception("Vivisect doesn't know how to handle a model of type '{}'".format(type(model)))


def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs, batch_size=32):
    for fw in fws:
        if isinstance(model, fw.model_types):
            fw.train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs, batch_size=batch_size)
            return True
    raise Exception("Vivisect doesn't know how to handle a model of type '{}'".format(type(model)))
