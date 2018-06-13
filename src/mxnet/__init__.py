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
import numpy
from urllib.request import urlopen, Request
import json
import logging


def probe(model, host, port, every=1, select=lambda x : True):
    assert(isinstance(model, Block))
    def callback(op, ivars, ovar):
        r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : ovar.asnumpy().tolist(),
                                                                                       "inputs" : [ivar.asnumpy().tolist() for ivar in ivars],
                                                                                       "type" : "LAYER",
                                                                                       "metadata" : {"name" : str(op).replace("\n", " "),
                                                                                                     "framework" : "mxnet",
                                                                                       },
        }).encode())
        urlopen(r)
    model.apply(lambda m : m.register_forward_hook(callback))

    
class BlockModel(Sequential):
    def __init__(self):
        super(BlockModel, self).__init__()
        self.add(Dense(20))
        self.add(Dense(3))

    
def block_mlp():
    return BlockModel()


def symbol_mlp():
    data = symbol.Variable("data")
    first_layer = symbol.FullyConnected(data=data, num_hidden=20)
    second_layer = symbol.FullyConnected(data=first_layer, num_hidden=3)
    return data, second_layer


def train(model, x_train, y_train, x_dev, y_dev, epochs):
    model.initialize()
    criterion = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    trainer = mxnet.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})

    x_train = nd.array(numpy.asfarray(x_train))
    y_train = nd.array(numpy.asfarray(y_train))
    x_dev = nd.array(numpy.asfarray(x_dev))
    y_dev = nd.array(numpy.asfarray(y_dev))
    
    for t in range(epochs):
        with mxnet.autograd.record():
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
        loss.backward()
        trainer.step(y_train.shape[0])
        logging.info("Train loss: {}".format(mxnet.nd.sum(loss).asscalar()))

    
