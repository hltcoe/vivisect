import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import Graph
from tensorflow.contrib.timeseries import ARModel
from tensorflow import keras
from tensorflow import nn
from tensorflow import layers
from tensorflow.contrib.rnn import LSTMCell
from types import MethodType
import json
from urllib.request import urlopen, Request
import numpy
import logging


model_types = (tf.Session)


def get_ops(model):
    return [(o.name, o) for o in model.graph.get_operations()]


def get_operation_names(model):
    return [o.name for o in model.graph.get_operations()]


def get_parameter_names(model):
    return list(model.collect_params().keys())
    return [name for name, _ in model.named_parameters()]    

    
def forward_attach(operation, callback):
    return None
    def _callback(op, inputs, outputs):
        logging.debug("Operation: {}".format(op._vivisect["op_name"]))
        return callback(op,                        
                        unpack_inputs(inputs, op),
                        unpack_outputs(outputs, op),
                        )
    operation.register_forward_hook(_callback)

    
def backward_attach(operation, callback):
    return None
    def _callback(op, grad_inputs, grad_outputs):
        logging.debug("Operation: {}".format(op._vivisect["op_name"]))
        return callback(op,                        
                        unpack_inputs(grad_inputs, op),
                        unpack_outputs(grad_outputs, op),
                        )
    operation.register_backward_hook(_callback)

    
def parameter_attach(model, callback):
    return None
    def _callback(op, inputs, outputs):
        logging.debug("Operation: {}".format(op._vivisect["op_name"]))
        params = op.named_parameters()
        return callback(op,                        
                        unpack_parameters(params, op),
                        )
    model.register_forward_hook(_callback)
        

def probe(model, host, port, select=lambda x : True, perform=lambda m, i, iv, ov : True):
    assert(isinstance(model, tf.Session))
    #def callback(self, *args, **argdict):
    #    retval = self.eval_(*args, **argdict)
    #    print(retval)
    #    return retval
    #model.run_ = model.run
    #model.run = MethodType(callback, model)
    # def callback(self, *args, **argdict):
    #     retval = self.run_(*args, **argdict)
    #     print(type(retval), 100)
    #     return retval
    #for op in model.graph.get_operations():
    #    for output in op.outputs:
    #        print(output)
    #        output.eval_ = output.eval
    #        output.eval = MethodType(callback, output)
    #     print(op)
    #     op.run_ = op.run
    #     op.run = MethodType(callback, op)

    #def callback(self, *args, **argdict):
    #    retval = self._run(*args, **argdict)
    #    print(argdict["feed_dict"])
    #    return retval    
    #for op in model.graph.get_operations():
    #    op._run = model.run
    #    op.run = MethodType(callback, op)
    #    pass


    
    def _run(self, *args, **argdict):
        retval = self.run_(*args, **argdict)
        for op in self.graph.get_operations():
            if select(op) and perform(model, op, [], []):
                metadata = {k : v for k, v in getattr(model, "_vivisect", {}).items()}
                metadata["op_name"] = str(op)
                r = Request("http://{}:{}".format(host, port),
                            headers={"Content-Type" : "application/json"},
                            method="POST",
                            data=json.dumps({"outputs" : [],
                                             "inputs" : [],
                                             "metadata" : metadata,
                            }).encode())
                urlopen(r)
        return retval    
    model.run_ = model.run
    model.run = MethodType(_run, model)


class mlp(tf.Session):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(mlp, self).__init__()
        self._nlabels = nlabels
        self._nfeats = nfeats
        self.input = tf.placeholder(tf.float32, shape=(None, nfeats), name="input")
        self.y = tf.placeholder(tf.float32, shape=(None, nlabels), name="gold")
        self.layer_one = layers.dense(inputs=self.input, units=hidden_size, name="layer1")
        self.output = layers.dense(inputs=self.layer_one, units=nlabels, name="output")


class rnn(tf.Session):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(rnn, self).__init__()
        self._nlabels = nlabels
        self._nfeats = nfeats
        self.input = (tf.placeholder(tf.float32, shape=(None, None, nfeats), name="data"), tf.placeholder(tf.int32, shape=(None,), name="lengths"))
        self.y = tf.placeholder(tf.int32, shape=(None, self._nlabels))
        cell = nn.rnn_cell.BasicRNNCell(hidden_size)
        _, self.state = nn.dynamic_rnn(cell, inputs=self.input[0], sequence_length=self.input[1], dtype=tf.float32)
        self.output = layers.dense(inputs=self.state, units=nlabels)


def onehot(v, r):
    retval = numpy.zeros((v.shape[0], r))
    for i, x in enumerate(v.tolist()):
        retval[i][x] = 1.0
    return retval

def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs, batch_size=32):
    model._vivisect["mode"] = "preinit"
    nlabels = model._nlabels
    nfeats = model._nfeats
    
    y_train, y_dev, y_test = [onehot(x, nlabels) for x in [y_train, y_dev, y_test]]
    
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.output,
                                                                     labels=model.y))
    train_op = optimizer.minimize(loss)
    model.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        model._vivisect["epoch"] += 1
        model._vivisect["mode"] = "train"
        _, train_loss = model.run([train_op, loss], feed_dict={model.input : x_train,
                                                               model.y : y_train})
        model._vivisect["mode"] = "dev"
        dev_loss = model.run(loss, feed_dict={model.input : x_dev,
                                              model.y : y_dev})
        model._vivisect["mode"] = "test"
        test_loss = model.run(loss, feed_dict={model.input : x_test,
                                               model.y : y_test})
        logging.info("Epoch {} train/dev/test loss: {:.4f}/{:.4f}/{:.4f}".format(epoch + 1, train_loss, dev_loss, test_loss))        
