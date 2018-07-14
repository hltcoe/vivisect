from urllib.request import urlopen, Request
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.utils.data
import torch
import numpy
import logging
import uuid
import functools
import re
import sys
import torch


model_types = (nn.Module)


def get_ops(model):
    for name, submodule in model.named_modules():
        yield((name, submodule))


def unpack_parameters(params, op):
    return {k : v.data.tolist() for k, v in params}


def unpack_outputs(outputs, op):
    if isinstance(outputs, torch.Tensor):
        return {"output" : outputs.data.tolist()}
    elif isinstance(op, nn.LSTM):
        o, (h, c) = outputs
        return {"hidden" : h.squeeze().data.tolist(),
                "state" : c.squeeze().data.tolist(),
                "output" : pad_packed_sequence(o, batch_first=True, total_length=300)[0].squeeze().data.tolist()
        }
    elif isinstance(outputs, tuple):
        return {str(i) : v.squeeze().data.tolist() for i, v in enumerate(outputs) if hasattr(v, "data")}
    else:
        raise Exception("Unknown output from {} (a {})".format(type(outputs), type(op)))


def unpack_inputs(inputs, op):
    return {}
    if isinstance(inputs, torch.Tensor):
        return {"input" : inputs.data.tolist()}
    elif isinstance(inputs, tuple):
        return {str(i) : v.squeeze().data.tolist() for i, v in enumerate(inputs) if hasattr(v, "data")}
    else:
        raise Exception("Unknown input to {} (a {})".format(type(inputs), type(op)))
    

def get_operation_names(model):
    return [name for name, _ in model.named_modules() if name != ""]


def get_parameter_names(model):
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
    def _callback(op, grad_inputs, grad_outputs):
        logging.debug("Operation: {}".format(op._vivisect["op_name"]))
        return callback(op,                        
                        unpack_inputs(grad_inputs, op),
                        unpack_outputs(grad_outputs, op),
                        )
    operation.register_backward_hook(_callback)

    
def parameter_attach(model, callback):
    def _callback(op, inputs, outputs):
        logging.debug("Operation: {}".format(op._vivisect["op_name"]))
        params = op.named_parameters()
        return callback(op,                        
                        unpack_parameters(params, op),
                        )
    model.register_forward_hook(_callback)
    

def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs, batch_size=32):
    def make_loader(vals):
        x, y = vals
        x = [torch.autograd.Variable(torch.from_numpy(numpy.asfarray(v))) for v in (x if isinstance(x, tuple) else [x])]
        y = [torch.autograd.Variable(torch.from_numpy(numpy.asarray(v))) for v in (y if isinstance(y, tuple) else [y])]
        data = torch.utils.data.TensorDataset(*x, *y)
        return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    x_size = len(x_train) if isinstance(x_train, (list, tuple)) else 1
    train_loader, dev_loader, test_loader = map(make_loader, [(x_train, y_train), (x_dev, y_dev), (x_test, y_test)])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    sm = nn.LogSoftmax(dim=1)
    model._vivisect["epoch"] = 0
    for t in range(epochs):
        model._vivisect["epoch"] += 1
        model._vivisect["mode"] = "train"
        train_loss = 0.0
        preds = []
        targs = []
        for i, batch in enumerate(train_loader, 1):
            y_pred = model(batch[0:x_size])
            preds += y_pred.argmax(1).data.tolist()
            #targs += 
            loss = criterion(y_pred, batch[-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.tolist()
        #cor = [x for x in zip(preds, y_train.tolist()) if x[0] == x[1]]
        #logging.info(len(cor) / len(preds))
        
        model._vivisect["mode"] = "dev"
        dev_loss = 0.0
        for i, batch in enumerate(dev_loader, 1):
            y_pred = model(batch[0:x_size])
            loss = criterion(y_pred, batch[-1])
            dev_loss += loss.data.tolist()

        model._vivisect["mode"] = "test"            
        test_loss = 0.0
        for i, batch in enumerate(test_loader, 1):
            y_pred = model(batch[0:x_size])
            loss = criterion(y_pred, batch[-1])
            test_loss += loss.data.tolist()
        logging.info("Iteration {} train/dev/test loss: {:.4f}/{:.4f}/{:.4f}".format(t + 1, train_loss, dev_loss, test_loss))

