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
    def _run(self, *args, **argdict):
        retval = self.run_(*args, **argdict)
        for op in self.graph.get_operations():
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : None,
                                                                                           "inputs" : None,
                                                                                           "type" : "LAYER",
                                                                                           "metadata" : {"name" : op.name,
                                                                                                         "framework" : "tensorflow",
                                                                                           },
            }).encode())
            urlopen(r)
        return retval
    
    sess.run_ = sess.run
    sess.run = MethodType(_run, sess)

    
def mlp():
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=(1000, 20), name="data")
    layer_one = layers.dense(inputs=x, units=20)
    layer_two = layers.dense(inputs=layer_one, units=3, name="output")
    return sess


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
