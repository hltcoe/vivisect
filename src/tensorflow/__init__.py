import argparse
import tensorflow as tf
#from vivisect.tensorflow import probe
from tensorflow import Graph
from tensorflow.contrib.timeseries import ARModel
from tensorflow import keras
#from tensorflow.keras import applications #import InceptionV3
#import tensorflow.contrib.keras as keras
from tensorflow import nn
from tensorflow import layers
from types import MethodType
import json
from urllib.request import urlopen, Request
#def _probe(model):
#    assert(isinstance(model, Graph))
#    for c in model.get_all_collection_keys():
#        print(c) #type(c))
#    print(type(model))

#        print(op.name)

def probe(sess, host, port):
    assert(isinstance(sess, tf.Session))
    def _run(self, *args, **argdict):
        self.graph.finalize()
        for op in self.graph.get_operations():
            r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"output" : op.name,
                                                                                           "inputs" : op.name,
                                                                                           "metadata" : op.name
            }).encode())
            urlopen(r)
    sess.run = MethodType(_run, sess)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=39628, type=int, help="Port number")
    args = parser.parse_args()
    
    with tf.Session() as sess:
        input_layer = tf.reshape([i for i in range(500)], [1, 500])
        layer_one = layers.dense(inputs=input_layer, units=100)
        layer_two = layers.dense(inputs=layer_one, units=2)
        
        probe(sess, args.host, args.port)        
        sess.run(1)

