import argparse
import tensorflow as tf
from tensorflow import Graph
from tensorflow.contrib.timeseries import ARModel
from tensorflow import keras
from tensorflow import nn
from tensorflow import layers
from types import MethodType
import json
from urllib.request import urlopen, Request
import numpy
import logging


def probe(sess, host, port, every=1):
    assert(isinstance(sess, tf.Session))
    #def callback(self, *args, **argdict):
    #    retval = self.eval_(*args, **argdict)
    #    print(retval)
    #    return retval
    #sess.run_ = sess.run
    #sess.run = MethodType(callback, sess)
    # def callback(self, *args, **argdict):
    #     retval = self.run_(*args, **argdict)
    #     print(type(retval), 100)
    #     return retval
    #for op in sess.graph.get_operations():
    #    for output in op.outputs:
    #        print(output)
    #        output.eval_ = output.eval
    #        output.eval = MethodType(callback, output)
    #     print(op)
    #     op.run_ = op.run
    #     op.run = MethodType(callback, op)
    def _run(self, *args, **argdict):
        retval = self.run_(*args, **argdict)
        for op in self.graph.get_operations():
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"outputs" : None,
                                                                                           "inputs" : None,
                                                                                           "op_name" : "LAYER",
                                                                                           "metadata" : {"name" : op.name,
                                                                                                         "framework" : "tensorflow",
                                                                                           },
            }).encode())
            urlopen(r)
        return retval
    
    sess.run_ = sess.run
    sess.run = MethodType(_run, sess)


class mlp(tf.Session):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(mlp, self).__init__()
        x = tf.placeholder(tf.float32, shape=(1000, nfeats), name="data")
        layer_one = layers.dense(inputs=x, units=hidden_size)
        layer_two = layers.dense(inputs=layer_one, units=nlabels, name="output")


def train(model, x_train, y_train, x_dev, y_dev, epochs):
    
    x_train = numpy.asfarray(x_train)
    y_train = tf.Variable(numpy.asfarray(y_train))
    x_dev = tf.Variable(numpy.asfarray(x_dev))
    y_dev = tf.Variable(numpy.asfarray(y_dev))
    
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.graph.get_tensor_by_name("output/BiasAdd:0"), 
                                                                     labels=y_train))
    train_op = optimizer.minimize(loss)
    model.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        _, train_loss = model.run([train_op, loss], feed_dict={model.graph.get_tensor_by_name("data:0") : x_train})
        logging.info("Train loss: {}".format(train_loss))
