from mxnet.gluon import Block, HybridBlock, SymbolBlock
from mxnet.symbol import Symbol, FullyConnected, Variable
from urllib.request import urlopen, Request
import json

def traverse(sym):
    return [sym] + map(traverse, sym.get_children())

def probe(model, host, port, every=1, select=lambda x : True):
    
    assert(isinstance(model, (Block, Symbol)))
    
    def callback(op, ivars, ovar):
        r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : ovar.asnumpy().tolist(),
                                                                                       "inputs" : [ivar.asnumpy().tolist() for ivar in ivars],
                                                                                       "metadata" : {"name" : str(op).replace("\n", " "),
                                                                                       },
        }).encode())
        urlopen(r)

    if isinstance(model, Block):
        model.apply(lambda m : m.register_forward_hook(callback))
    else:
        print(dir(model))
