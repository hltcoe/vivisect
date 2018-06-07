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
    
    train_class_probs = numpy.random.dirichlet([1.0 for i in range(3)])
    dev_class_probs = numpy.random.dirichlet([1.0 for i in range(3)])
    obs_probs = numpy.random.dirichlet([1.0 for i in range(20)], size=3)
    train_true_classes = numpy.random.multinomial(1, train_class_probs, size=1000)
    dev_true_classes = numpy.random.multinomial(1, dev_class_probs, size=100)
    train_obs = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in train_true_classes.argmax(1)]).squeeze()
    dev_obs = numpy.asfarray([numpy.random.multinomial(10, obs_probs[c, :], size=1) for c in dev_true_classes.argmax(1)]).squeeze()

    
    # PyTorch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    torch.set_default_tensor_type("torch.DoubleTensor")
    from vivisect.pytorch import probe
    class PyTorchModel(nn.Module):
        def __init__(self):
            super(PyTorchModel, self).__init__()
            self.dense1 = nn.Linear(in_features=20, out_features=20)
            self.dense2 = nn.Linear(in_features=20, out_features=3)        
        def forward(self, x):
            x = F.relu(self.dense1(x))
            return F.relu(self.dense2(x))

    x = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(train_obs)))
    y = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(train_true_classes)))
    x_dev = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(dev_obs)))
    y_dev = torch.autograd.Variable(torch.from_numpy(numpy.asfarray(dev_true_classes)))

    logging.info("Testing with PyTorch model...")
    model = PyTorchModel()
    probe(model, args.host, args.port, every=3)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(args.epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_dev_pred = model(x_dev)
        dev_loss = criterion(y_dev_pred, y_dev)
        logging.info("Dev loss: {}".format(dev_loss.data.tolist()[0]))    
        
        
    # MXNet
    import mxnet
    v1, v2, v3 = map(int, mxnet.__version__.split("."))
    if v1 != 1 or v2 < 3:
        logging.info("Your MXNet version is {}, but Vivisection requires >=1.3.0: you may need to install from source.  Skipping MXNet for now...".format(mxnet.__version__))
    else:
        logging.info("Testing with MXNet model...")
        from mxnet import nd
        from mxnet.gluon.nn import Sequential, Dense
        from mxnet.gluon import Block, HybridBlock, SymbolBlock, Trainer
        from vivisect.mxnet import probe
        class MXNetModel(Sequential):
            def __init__(self):
                super(MXNetModel, self).__init__()
                self.add(Dense(20))
                self.add(Dense(3))

        x = nd.array(numpy.asfarray(train_obs))
        y = nd.array(numpy.asfarray(train_true_classes))
        x_dev = nd.array(numpy.asfarray(dev_obs))
        y_dev = nd.array(numpy.asfarray(dev_true_classes))

        model = MXNetModel()
        probe(model, args.host, args.port, every=3)
        model.initialize()

        criterion = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        trainer = mxnet.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})

        for t in range(args.epochs):
            with mxnet.autograd.record():
                y_pred = model(x)
                loss = criterion(y_pred, y)
            loss.backward()
            trainer.step(y.shape[0])
            y_dev_pred = model(x_dev)
            loss = criterion(y_dev_pred, y_dev)
            dev_loss = mxnet.nd.sum(loss).asscalar()
            logging.info("Dev loss: {}".format(dev_loss))


    # Tensorflow
    logging.info("Testing with Tensorflow model...")    
    from vivisect.tensorflow import probe
    import tensorflow as tf
    from tensorflow import layers
    with tf.Session() as sess:
        input_layer = tf.reshape([i for i in range(20)], [1, 20])
        layer_one = layers.dense(inputs=input_layer, units=20)
        layer_two = layers.dense(inputs=layer_one, units=3)        
        probe(sess, args.host, args.port)        
        sess.run(1)
            
        
