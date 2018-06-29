import argparse
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
        self.x = tf.placeholder(tf.float32, shape=(None, nfeats), name="data")
        self.y = tf.placeholder(tf.float32, shape=(None, nlabels), name="gold")
        self.layer_one = layers.dense(inputs=self.x, units=hidden_size, name="layer1")
        self.layer_two = layers.dense(inputs=self.layer_one, units=nlabels, name="layer2")


class rnn(tf.Session):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(rnn, self).__init__()
        self._nlabels = nlabels
        self._nfeats = nfeats
        #batch_size = 32
        self.x = tf.placeholder(tf.float32, shape=(None, None, nfeats), name="data")
        self.lengths = tf.placeholder(tf.int32, shape=(None,), name="lengths")
        self.y = tf.placeholder(tf.int32, shape=(None,))
        self.y_oh = tf.one_hot(self.y, depth=self._nlabels)
        cell = nn.rnn_cell.BasicRNNCell(hidden_size)
        #initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        _, self.state = nn.dynamic_rnn(cell, inputs=self.x, sequence_length=self.lengths, dtype=tf.float32) #, initial_state=initial_state)
        
        self.layer_two = layers.dense(inputs=self.state, units=nlabels)
        #self.dense = layers.dense(inputs=self.lstm, units=nlabels, name="dense")


def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs, batch_size=32):
    model._vivisect["mode"] = "preinit"
    nlabels = model._nlabels
    nfeats = model._nfeats
    
    #x_size = len(x_train) if isinstance(x_train, (list, tuple)) else 1
    #x_train = numpy.asfarray(x_train)
    #y_train = numpy.asfarray([[(1 if i == y else 0) for i in range(nlabels)] for y in y_train])
    #x_dev = numpy.asfarray(x_dev)
    #y_dev = numpy.asfarray([[(1 if i == y else 0) for i in range(nlabels)] for y in y_dev])
    #x_test = numpy.asfarray(x_test)
    #y_test = numpy.asfarray([[(1 if i == y else 0) for i in range(nlabels)] for y in y_test])
    
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.layer_two,
                                                                     labels=model.y_oh))
    train_op = optimizer.minimize(loss)
    model.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        model._vivisect["iteration"] += 1
        model._vivisect["mode"] = "train"
        _, train_loss = model.run([train_op, loss], feed_dict={model.x : x_train[0],
                                                               model.lengths : x_train[1],
                                                               model.y : y_train})
        model._vivisect["mode"] = "dev"
        dev_loss = model.run(loss, feed_dict={model.x : x_dev,
                                              model.y : y_dev})
        model._vivisect["mode"] = "test"
        test_loss = model.run(loss, feed_dict={model.x : x_test,
                                               model.y : y_test})
        logging.info("Iteration {} train/dev/test loss: {:.4f}/{:.4f}/{:.4f}".format(epoch + 1, train_loss, dev_loss, test_loss))        
