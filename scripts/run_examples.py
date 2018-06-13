import argparse
import numpy
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=39628, type=int, help="Port number")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="Maximum training epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    
    # generate some synthetic data from a mixture model
    train_class_probs = numpy.random.dirichlet([1.0 for i in range(3)])
    dev_class_probs = numpy.random.dirichlet([1.0 for i in range(3)])
    obs_probs = numpy.random.dirichlet([1.0 for i in range(20)], size=3)
    y_train = numpy.random.multinomial(1, train_class_probs, size=1000)
    y_dev = numpy.random.multinomial(1, dev_class_probs, size=100)
    x_train = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in y_train.argmax(1)]).squeeze()
    x_dev = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in y_dev.argmax(1)]).squeeze()

    
    # Tensorflow
    import tensorflow
    from vivisect.tensorflow import probe, mlp, train

    logging.info("Testing with Tensorflow 'Session'...")
    model = mlp()
    assert(isinstance(model, tensorflow.Session))
    probe(model, args.host, args.port, every=1)
    train(model, x_train, y_train, x_dev, y_dev, args.epochs)

    
    # PyTorch
    import torch
    from vivisect.pytorch import probe, mlp, train

    logging.info("Testing with PyTorch 'Module'...")
    model = mlp()
    assert(isinstance(model, torch.nn.Module))
    probe(model, args.host, args.port, every=1)
    train(model, x_train, y_train, x_dev, y_dev, args.epochs)

        
    # MXNet
    import mxnet
    from vivisect.mxnet import probe, block_mlp, symbol_mlp, train
    from mxnet.gluon import Block, HybridBlock, SymbolBlock, Trainer

    logging.info("Testing with MXNet 'Symbol'...")
    input, output = symbol_mlp()
    assert(isinstance(input, mxnet.sym.Symbol) and isinstance(output, mxnet.sym.Symbol))
    model = SymbolBlock(output, input)
    probe(model, args.host, args.port, every=1)
    train(model, x_train, y_train, x_dev, y_dev, args.epochs)
    
    logging.info("Testing with MXNet 'Block'...")
    model = block_mlp()
    assert(isinstance(model, mxnet.gluon.Block))
    probe(model, args.host, args.port, every=1)
    train(model, x_train, y_train, x_dev, y_dev, args.epochs)
