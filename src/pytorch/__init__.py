import torch.nn as nn
from urllib.request import urlopen, Request
import json

def probe(model, host, port, every=1, select=lambda x : True):
    
    assert(isinstance(model, nn.Module))
    
    def callback(op, ivars, ovar):
        inum = getattr(op, "iteration", 1)
        op.iteration = inum + 1
        if inum % every == 0:
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : ovar.data.tolist(),
                                                                                           "inputs" : [ivar.data.tolist() for ivar in ivars],
                                                                                           "metadata" : {"name" : str(op),
                                                                                                         "iteration" : inum,
                                                                                           }
            }).encode())
            urlopen(r)
        
        
    for child in model.children():
        probe(child, host, port)
        if select(child):
            child.register_forward_hook(callback) 


