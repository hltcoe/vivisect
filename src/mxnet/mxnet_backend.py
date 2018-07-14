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


model_types = (Block)


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

    x_train = nd.array(x_train)
    y_train = nd.array(y_train)
    x_dev = nd.array(x_dev)
    y_dev = nd.array(y_dev)
    x_test = nd.array(x_test)
    y_test = nd.array(y_test)
    
    criterion = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    trainer = mxnet.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': .1})
    
    for t in range(epochs):
        model._vivisect["epoch"] += 1
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
        
        logging.info("Iteration {} train/dev/test loss: {:.4f}/{:.4f}/{:.4f}".format(t + 1, train_loss, dev_loss, test_loss))
    
