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
from mxnet.gluon.data import DataLoader, ArrayDataset
import numpy
from urllib.request import urlopen, Request
import json
import logging
from types import MethodType


model_types = (Block)
framework_name = "Gluon"


def get_ops(model):
    return [(o.name, o) for o in _get_ops(model)]


def _get_ops(model):
    return [model] + sum([_get_ops(c) for n, c in model._children.items()], [])


def unpack_parameters(params, op):
    return {k : v.data.tolist() for k, v in params}


def unpack_outputs(outputs, op):
    #if isinstance(outputs, torch.Tensor):
    return {"output" : outputs.asnumpy().tolist()}
    # elif isinstance(op, nn.LSTM):
    #     o, (h, c) = outputs
    #     return {"hidden" : h.squeeze().data.tolist(),
    #             "state" : c.squeeze().data.tolist(),
    #             "output" : pad_packed_sequence(o, batch_first=True, total_length=300)[0].squeeze().data.tolist()
    #     }
    # elif isinstance(outputs, tuple):
    #     return {str(i) : v.squeeze().data.tolist() for i, v in enumerate(outputs) if hasattr(v, "data")}
    # else:
    #     raise Exception("Unknown output from {} (a {})".format(type(outputs), type(op)))


def unpack_inputs(inputs, op):
    return {}
    if isinstance(inputs, torch.Tensor):
        return {"input" : inputs.data.tolist()}
    elif isinstance(inputs, tuple):
        return {str(i) : v.squeeze().data.tolist() for i, v in enumerate(inputs) if hasattr(v, "data")}
    else:
        raise Exception("Unknown input to {} (a {})".format(type(inputs), type(op)))
    

def get_operation_names(model):
    return [x.name for x in _get_ops(model)]


def get_parameter_names(model):
    return list(model.collect_params().keys())
    return [name for name, _ in model.named_parameters()]    

    
def forward_attach(operation, callback):
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


def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs):
    model.initialize()
    x_train = x_train[0] if isinstance(x_train, (list, tuple)) else x_train
    x_dev = x_dev[0] if isinstance(x_dev, (list, tuple)) else x_dev
    x_test = x_test[0] if isinstance(x_test, (list, tuple)) else x_test
    x_train = nd.array(x_train)
    y_train = nd.array(y_train)
    x_dev = nd.array(x_dev)
    y_dev = nd.array(y_dev)
    x_test = nd.array(x_test)
    y_test = nd.array(y_test)
    
    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=32)
    dev_loader = DataLoader(ArrayDataset(x_dev, y_dev), batch_size=32)
    test_loader = DataLoader(ArrayDataset(x_test, y_test), batch_size=32)
    
    criterion = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    trainer = mxnet.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': .1})
    
    for t in range(epochs):
        model._vivisect["epoch"] += 1
        model._vivisect["mode"] = "train"
        train_loss, dev_loss, test_loss = 0.0, 0.0, 0.0
        for x, y in train_loader:
            with mxnet.autograd.record():            
                y_pred = model(x)
                loss = criterion(y_pred, y)
            loss.backward()
            trainer.step(y_train.shape[0])
            train_loss += mxnet.nd.sum(loss).asscalar()
        
        model._vivisect["mode"] = "dev"
        for x, y in dev_loader:
            with mxnet.autograd.record():            
                y_pred = model(x)
                loss = criterion(y_pred, y)
            dev_loss += mxnet.nd.sum(loss).asscalar()

        model._vivisect["mode"] = "test"
        for x, y in test_loader:
            with mxnet.autograd.record():            
                y_pred = model(x)
                loss = criterion(y_pred, y)
            test_loss += mxnet.nd.sum(loss).asscalar()
        
        logging.info("Iteration {} train/dev/test loss: {:.4f}/{:.4f}/{:.4f}".format(t + 1, train_loss, dev_loss, test_loss))
    
