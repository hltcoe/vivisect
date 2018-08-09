#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy
import warnings
import logging
import gzip
import random
import os
from vivisect import probe, train, register_classification_targets, register_clustering_targets, flush, clear, remove
warnings.simplefilter(action='ignore', category=FutureWarning)


def onehot(i, r):
    retval = [0] * r
    retval[i] = 1.0
    return retval

           
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", default="MLP", help="Model name")
    parser.add_argument("--host", dest="host", default="aggregator", help="Host name")
    parser.add_argument("--port", dest="port", default=8080, type=int, help="Port number")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="Maximum training epochs")
    parser.add_argument("--hidden_size", dest="hidden_size", default=40, type=int, help="Hidden size for MLPs/LSTMs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    # generate some synthetic data from a mixture model
    n_points = 1000
    n_mlp_feats = 20
    n_mlp_labels = 3
    train_class_probs = numpy.random.dirichlet([1.0 for i in range(n_mlp_labels)])
    dev_class_probs = numpy.random.dirichlet([1.0 for i in range(n_mlp_labels)])
    test_class_probs = numpy.random.dirichlet([1.0 for i in range(n_mlp_labels)])
    obs_probs = numpy.random.dirichlet([1.0 for i in range(n_mlp_feats)], size=n_mlp_labels)
    y_train = numpy.asarray([random.randint(0, n_mlp_labels - 1) for _ in range(n_points)], dtype=numpy.int64)
    y_dev = numpy.asarray([random.randint(0, n_mlp_labels - 1) for _ in range(int(.1 * n_points))], dtype=numpy.int64)
    y_test = numpy.asarray([random.randint(0, n_mlp_labels - 1) for _ in range(int(.1 * n_points))], dtype=numpy.int64)
    x_train = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in y_train]).squeeze()
    x_dev = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in y_dev]).squeeze()
    x_test = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in y_test]).squeeze()
    

    def which_operation(model, operation):
        return True #(operation._v.operation_name == "softmax")

    def which_array(model, operation, array_info):
        return True
    
    def when(model, operation=None, array=None):
        return (model._v.state == "train")

    
    # logging.info("Testing with Tensorflow 'Session'")
    # import tensorflow
    # from vivisect.tensorflow import mlp
    
    # logging.info("Tensorflow MLP model")    
    # model = mlp(n_mlp_feats, n_mlp_labels, args.hidden_size)
    # model._vivisect = {"epoch" : 0, "model_name" : "Tensorflow MLP Model", "framework" : "tensorflow"}
    # assert(isinstance(model, tensorflow.Session))
    # probe(model, args.host, args.port, which, when, model_name="Tensorflow MLP")
    # train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, args.epochs)
    

    import torch    
    from vivisect.pytorch import mlp

    logging.info("PyTorch MLP model")    
    model = mlp(n_mlp_feats, n_mlp_labels, args.hidden_size)    
    assert(isinstance(model, torch.nn.Module))
    remove(args.host, args.port, "MLP")
    probe(args.name, model, args.host, args.port, which_operation, when)
    register_classification_targets(args.host, args.port, name="Classify components", targets=y_train, model_pattern="MLP")
    register_clustering_targets(args.host, args.port, name="Cluster components", targets=y_train, model_pattern="MLP")
    train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, args.epochs, batch_size=10)


    # from mxnet.gluon import Block, HybridBlock, SymbolBlock, Trainer
    # from vivisect.mxnet import mlp

    # logging.info("Gluon MLP model")
    # model = mlp(n_mlp_feats, n_mlp_labels, args.hidden_size)    
    # #model._vivisect = {"epoch" : 0, "framework" : "pytorch"}
    # assert(isinstance(model, Block))
    # #probe(model, args.host, args.port, which, when, model_name="Gluon MLP")
    # logging.info("Operations: %s, Parameters: %s", *get_model_info(model))
    # register_classification_targets(args.host, args.port, name="Classify", targets=y_train, model_pattern="Gluon MLP") #, layer_pattern=".*outputs.*")
    # register_clustering_targets(args.host, args.port, name="Cluster", targets=y_train, model_pattern="Gluon MLP") #, layer_pattern=".*outputs.*")
    # train(model, x_train, y_train, x_dev, y_dev, x_test, y_test, args.epochs)

    
    flush(args.host, args.port)
