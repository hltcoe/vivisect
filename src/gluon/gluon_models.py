from mxnet.gluon import Block, HybridBlock, SymbolBlock
from mxnet.symbol import Symbol, FullyConnected, Variable
import mxnet
from mxnet import symbol
from mxnet import nd
from mxnet import module
from mxnet import init
from mxnet import io
from mxnet.gluon import Block, HybridBlock, SymbolBlock, Trainer
from mxnet.gluon.nn import Sequential, Dense
from mxnet.gluon.rnn import LSTM
import numpy
from urllib.request import urlopen, Request
import json
import logging
from types import MethodType

    
class mlp(Sequential):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(mlp, self).__init__()
        self.add(Dense(hidden_size, in_units=nfeats, activation="relu"))
        self.add(Dense(nlabels, in_units=hidden_size, activation="relu"))


class rnn(Sequential):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(rnn, self).__init__()
        self.add(LSTM(hidden_size, layout="NTC"))
        self.add(Dense(nlabels))
