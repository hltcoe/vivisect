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
