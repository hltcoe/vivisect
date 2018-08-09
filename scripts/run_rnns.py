#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy
import warnings
import logging
import gzip
import random
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#from vivisect.servers import flush, clear
from vivisect import probe, train, register_classification_targets, register_clustering_targets, flush, clear, remove
warnings.simplefilter(action='ignore', category=FutureWarning)


def onehot(i, r):
    retval = [0] * r
    retval[i] = 1.0
    return retval

           
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", default="RNN", help="Model name")
    parser.add_argument("--host", dest="host", default="aggregator", help="Host name")
    parser.add_argument("--port", dest="port", default=8080, type=int, help="Port number")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="Maximum training epochs")
    parser.add_argument("--hidden_size", dest="hidden_size", default=50, type=int, help="Hidden size for MLPs/LSTMs")
    parser.add_argument("--input", dest="input", default="/tmp/vivisect/data/lid.txt.gz", help="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    # read some data for an RNN model
    instances = []
    char_lookup = {"<S>" : 0, "</S>" : 1}
    label_lookup = {}
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            label, text = line.strip().split("\t")
            instances.append((label_lookup.setdefault(label, len(label_lookup)), [0] + [char_lookup.setdefault(c, len(char_lookup)) for c in text] + [1]))

    
    random.shuffle(instances)
    instances = instances[0:1000]
    max_len = max([len(x) for _, x in instances])
    train_instances = instances[0:int(.8 * len(instances))]
    y_rnn_train = numpy.asarray([l for l, _ in train_instances])
    lengths_rnn_train = numpy.asarray([len(cs) for _, cs in train_instances])
    x_rnn_train = numpy.asarray([[onehot(x, len(char_lookup)) for x in xs] + ([[0] * len(char_lookup)] * (max_len - len(xs))) for _, xs in train_instances])
    dev_instances = instances[int(.8 * len(instances)) : int(.9 * len(instances))]    
    y_rnn_dev = numpy.asarray([l for l, _ in dev_instances])
    lengths_rnn_dev = numpy.asarray([len(cs) for _, cs in dev_instances])
    x_rnn_dev = numpy.asarray([[onehot(x, len(char_lookup)) for x in xs] + ([[0] * len(char_lookup)] * (max_len - len(xs))) for _, xs in dev_instances])
    test_instances = instances[int(.9 * len(instances)):]
    y_rnn_test = numpy.asarray([l for l, _ in test_instances])
    lengths_rnn_test = numpy.asarray([len(cs) for _, cs in test_instances])
    x_rnn_test = numpy.asarray([[onehot(x, len(char_lookup)) for x in xs] + ([[0] * len(char_lookup)] * (max_len - len(xs))) for _, xs in test_instances])
    n_rnn_labels = len(label_lookup)
    n_rnn_feats = len(char_lookup)
    rlabel_lookup = {v : k for k, v in label_lookup.items()}
    rchar_lookup = {v : k for k, v in char_lookup.items()}

    
    #def which(model, operation):
    #    return (operation._vivisect["op_name"] != "")

    #def when(model, operation):
    #    return (model._vivisect["epoch"] % 1 == 0 and model._vivisect["mode"] == "train")
    def which_operation(model, operation):
        return True #(operation._v.name != "")

    def which_array(model, operation, array_info):
        return True
    
    def when(model, operation=None, array=None):
        return (model._v.epoch % 5 == 0 and model._v.state == "train")

    
    # logging.info("Testing with Tensorflow 'Session'")
    # import tensorflow
    # from vivisect.tensorflow import rnn

    # logging.info("RNN model")
    # model = rnn(n_rnn_feats, n_rnn_labels, args.hidden_size)
    # model._vivisect = {"iteration" : 0, "model_name" : "Tensorflow RNNe Model", "framework" : "tensorflow"}
    # assert(isinstance(model, tensorflow.Session))
    # probe(model, args.host, args.port)
    # train(model, (x_rnn_train, lengths_rnn_train), y_rnn_train, (x_rnn_dev, lengths_rnn_dev), y_rnn_dev, (x_rnn_test, lengths_rnn_test), y_rnn_test, args.epochs)
    

    import torch    
    from vivisect.pytorch import rnn
    
    logging.info("PyTorch RNN model")
    model = rnn(n_rnn_feats, n_rnn_labels, args.hidden_size, rlabel_lookup, rchar_lookup)
    assert(isinstance(model, torch.nn.Module))
    remove(args.host, args.port, "RNN")
    probe(args.name, model, args.host, args.port, which_operation, when)
    register_classification_targets(args.host, args.port, name="Classify language", targets=y_rnn_train, model_pattern="RNN")
    register_clustering_targets(args.host, args.port, name="Cluster language", targets=y_rnn_train, model_pattern="RNN")
    train(model, (x_rnn_train, lengths_rnn_train), y_rnn_train, (x_rnn_dev, lengths_rnn_dev), y_rnn_dev, (x_rnn_test, lengths_rnn_test), y_rnn_test, args.epochs, batch_size=2)

    
    # from vivisect.gluon import rnn
    # from mxnet.gluon import Block, HybridBlock, SymbolBlock, Trainer
    
    # logging.info("Gluon RNN model")
    # model = rnn(n_rnn_feats, n_rnn_labels, args.hidden_size)
    # model._vivisect = {"epoch" : 0, "framework" : "mxnet"}    
    # assert(isinstance(model, Block))
    # probe(model, args.host, args.port, which, when, model_name="Gluon RNN")
    # register_classification_targets(args.host, args.port, name="Classify", targets=y_rnn_dev, model_pattern="Gluon RNN") #, layer_pattern=".*outputs.*")
    # register_clustering_targets(args.host, args.port, name="Cluster", targets=y_rnn_dev, model_pattern="Gluon RNN") #, layer_pattern=".*outputs.*")
    # train(model, (x_rnn_train, lengths_rnn_train), y_rnn_train, (x_rnn_dev, lengths_rnn_dev), y_rnn_dev, (x_rnn_test, lengths_rnn_test), y_rnn_test, args.epochs)

    
    flush(args.host, args.port)
