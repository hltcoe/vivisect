from urllib.request import urlopen, Request
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data
import torch
import numpy
import logging
import uuid
torch.set_default_tensor_type("torch.DoubleTensor")


def flatten(tensors):
    if isinstance(tensors, (tuple, list)):
        return sum(map(flatten, tensors), [])
    else:
        return [tensors]


def probe(model, host, port, select=lambda x : True, do_send=lambda m, i, iv, ov : True):
    assert(isinstance(model, nn.Module))
    def callback(op, ivars, ovars):
        iteration = model._vivisect["iteration"]
        if do_send(model, op, ivars, ovars):
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : [v.data.tolist() for v in flatten(ovars)],
                                                                                           "inputs" : [ivar.data.tolist() for ivar in ivars],
                                                                                           "op_name" : str(op),
                                                                                           "metadata" : getattr(model, "_vivisect", {}),
            }).encode())
            urlopen(r)
            
    for child in model.children():
        probe(child, host, port)
        if select(child):
            child.register_forward_hook(callback) 


class mlp(nn.Module):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(mlp, self).__init__()
        self.dense1 = nn.Linear(in_features=nfeats, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=nlabels)
    def forward(self, x):
        x = F.relu(self.dense1(x[0]))
        return F.relu(self.dense2(x))

    
class rnn(nn.Module):
    def __init__(self, nfeats, nlabels, hidden_size):
        super(rnn, self).__init__()
        self.lstm = nn.LSTM(input_size=nfeats, hidden_size=hidden_size)
        self.dense = nn.Linear(in_features=hidden_size, out_features=nlabels)
    def forward(self, x):
        x, l = x
        seq_lengths, perm_idx = l.sort(0, descending=True)
        xp = pack_padded_sequence(x[perm_idx], seq_lengths.tolist(), batch_first=True)
        packed_out, (ht, ct) = self.lstm(xp)
        out, _ = pad_packed_sequence(packed_out)
        return self.dense(ht[-1])


def train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, epochs, batch_size=32):
    def make_loader(vals):
        x, y = vals
        x = [torch.autograd.Variable(torch.from_numpy(numpy.asfarray(v))) for v in (x if isinstance(x, tuple) else [x])]
        y = [torch.autograd.Variable(torch.from_numpy(numpy.asfarray(v))) for v in (y if isinstance(y, tuple) else [y])]
        data = torch.utils.data.TensorDataset(*x, *y)
        return torch.utils.data.DataLoader(data, batch_size=batch_size)
    
    x_size = len(x_train) if isinstance(x_train, (list, tuple)) else 1
    train_loader, dev_loader, test_loader = map(make_loader, [(x_train, y_train), (x_dev, y_dev), (x_test, y_test)])
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(epochs):
        model._vivisect["iteration"] += 1
        model._vivisect["mode"] = "train"
        train_loss = 0.0
        for i, batch in enumerate(train_loader, 1):
            y_pred = model(batch[0:x_size])
            loss = criterion(y_pred, batch[-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.tolist()

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
        logging.info("Train/dev/test loss: {}/{}/{}".format(train_loss, dev_loss, test_loss))        
