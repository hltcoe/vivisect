from urllib.request import urlopen, Request
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy
import logging
torch.set_default_tensor_type("torch.DoubleTensor")


def probe(model, host, port, every=1, select=lambda x : True):
    
    assert(isinstance(model, nn.Module))
    
    def callback(op, ivars, ovar):
        inum = getattr(op, "iteration", 1)
        op.iteration = inum + 1
        if inum % every == 0:
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : ovar.data.tolist(),
                                                                                           "inputs" : [ivar.data.tolist() for ivar in ivars],
                                                                                           "type" : "LAYER",
                                                                                           "metadata" : {"name" : str(op),
                                                                                                         "iteration" : inum,
                                                                                                         "framework" : "pytorch",
                                                                                           }
            }).encode())
            urlopen(r)
        
        
    for child in model.children():
        probe(child, host, port)
        if select(child):
            child.register_forward_hook(callback) 


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.dense1 = nn.Linear(in_features=20, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=3)        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        return F.relu(self.dense2(x))
    

def train(model, x_train, y_train, x_dev, y_dev, epochs):
    x_train = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(x_train)))
    y_train = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(y_train)))
    x_dev = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(x_dev)))
    y_dev = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(y_dev)))
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info("Train loss: {}".format(loss.data.tolist()[0]))    
