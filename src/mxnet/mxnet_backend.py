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
from sockeye.training import TrainingModel
import numpy
from urllib.request import urlopen, Request
import json
import logging
from types import MethodType


def probe(model, host, port, select=lambda x : True, perform=lambda m, i, iv, ov : True):
    assert(isinstance(model, (Block, mxnet.module.BaseModule)))
    if isinstance(model, Block):
        def callback(op, ivars, ovars):
            if perform(model, op, ivars, ovars):
                metadata = {k : v for k, v in getattr(model, "_vivisect", {}).items()}
                metadata["op_name"] = op.name
                r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"outputs" : [ovar.asnumpy().tolist() for ovar in (ovars if isinstance(ovars, list) else [ovars])],
                                                                                               "inputs" : [ivar.asnumpy().tolist() for ivar in ivars],
                                                                                               "metadata" : metadata,
                }).encode())
                urlopen(r)
        def register(m):
            if select(m):
                m.register_forward_hook(callback)
        model.apply(register)
    elif isinstance(model, mxnet.module.BaseModule):
        def callback(self, *args, **argdict):
            retval = self.forward_(*args, **argdict)
            metadata = {k : v for k, v in getattr(model, "_vivisect", {}).items()}
            metadata["op_name"] = self._symbol.name
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"outputs" : [o.asnumpy().tolist() for o in retval],
                                                                                           "inputs" : [],
                                                                                           "metadata" : metadata,
            }).encode())
            urlopen(r)


            #print(retval)
            #print(self.output_dict) #retval)
            #return retval

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

    
class mlp(Sequential):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(mlp, self).__init__()
        self.add(Dense(hidden_size))
        self.add(Dense(nlabels))


class rnn(Sequential):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(rnn, self).__init__()
        self.add(LSTM(hidden_size))
        self.add(Dense(nlabels))


def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs):
    model.initialize()
    criterion = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    trainer = mxnet.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})

    x_train = nd.array(x_train)
    y_train = nd.array(y_train)
    x_dev = nd.array(x_dev)
    y_dev = nd.array(y_dev)
    x_test = nd.array(x_test)
    y_test = nd.array(y_test)
    
    for t in range(epochs):
        model._vivisect["iteration"] += 1
        model._vivisect["mode"] = "train"
        with mxnet.autograd.record():
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
        loss.backward()
        trainer.step(y_train.shape[0])
        train_loss = mxnet.nd.sum(loss).asscalar()

        model._vivisect["mode"] = "dev"
        with mxnet.autograd.record():
            y_pred = model(x_dev)
            loss = criterion(y_pred, y_dev)
        dev_loss = mxnet.nd.sum(loss).asscalar()

        model._vivisect["mode"] = "test"
        with mxnet.autograd.record():
            y_pred = model(x_test)
            loss = criterion(y_pred, y_test)
        test_loss = mxnet.nd.sum(loss).asscalar()
        
        logging.info("Train/dev/test loss: {}/{}/{}".format(train_loss, dev_loss, test_loss))                

    
