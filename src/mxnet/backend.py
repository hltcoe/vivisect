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
from sockeye.training import TrainingModel
import numpy
from urllib.request import urlopen, Request
import json
import logging
from types import MethodType









def probe(model, host, port, every=1, select=lambda x : True):
    assert(isinstance(model, (Block, mxnet.module.BaseModule)))
    if isinstance(model, Block):
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
    elif isinstance(model, mxnet.module.BaseModule):
        def callback(self, *args, **argdict):
            retval = self.forward_(*args, **argdict)
            print(retval)
            #print(self.output_dict) #retval)
            return retval

        class Monitor:
            def install(self, exe):
                #def callback(name, handle):
                #    print(exe.forward())
                    #print(name) #, exe._get_outputs())
                    #o = exe.output_dict #get_outputs()
                    #print(o.keys())
                    #print(name)
                #print(exe)
                #exe.set_monitor_callback(lambda n, v : print(n, mxnet.nd.NDArray(mxnet.base.NDArrayHandle(v)).shape))
                #exe.set_monitor_callback(callback)
                exe.forward_ = exe.forward
                exe.forward = MethodType(callback, exe)
                #print(101)
        model.install_monitor(Monitor())
        # def callback(self, *args, **argdict):
        #     retval = self.forward_(*args, **argdict)
        #     for layers in model.get_params():
        #         for k, v in layers.items():
        #             print(k, v.shape)
        #     return retval
        #model.forward_ = model.forward
        #model.forward = MethodType(callback, model)
        # class Monitor:
        #    def __init__(self, _model):
        #        self._model = _model
        #    def tic(self):
        #        return []
        #    def toc(self):
        #        par, aux = self._model.module.get_params()
        #        r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : len(par),
        #                                                                                       "inputs" : len(aux),
        #                                                                                       "type" : "LAYER",
        #                                                                                       "metadata" : {#"name" : str(op).replace("\n", " "),
        #                                                                                                     "framework" : "sockeye",
        #                                                                                       },
        #        }).encode())
        #        urlopen(r)
        #    def install(self, x):
        #        pass
        # model._monitor = Monitor(model)

    
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

    
