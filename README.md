# Vivisect

This library is intended as unified task-based introspection of neural model layers for Tensorflow, PyTorch, and MXNet.  Minimally, it consists of two components: 

1.  A server, which provides a REST endpoint for receiving tensors and labels which it uses for lightweight tasks that calculate a *score* associated with each tensor's performance on each task
2.  A function, `probe`, imported from the `vivisect` library, that takes a neural model and attaches callbacks to its tensors for shipping them, along with appropriate labels, to the server, at appropriate intervals

It will support three types of *score*:

1.  Intrinsic scalar properties of tensors (e.g. mean, variance)
2.  Performance on supervised classification of labels based on tensor values (e.g. f-score using logistic regression)
3.  Unsupervised clustering based on tensor values (e.g. mutual information with a gold standard, using k-means)

and three popular deep learning frameworks, with associated model classes:

1.  Tensorflow *Session*
2.  PyTorch *Module*
3.  MXNet *Symbol* and *Block*

Libraries built on these frameworks should be able to use *vivisect* without modification if they subclass appropriately.

## Quick setup

Install the library and the supported frameworks:

```python
pip install . --user
```

*Note: Vivisect requires an MXNet version > 1.2 which includes the ability to attach hooks: until then, you'll need to [install from source](www.mxnet.com) or use PyTorch/Tensorflow*

Vivisect is composed of three servers that need to be running simultaneously, e.g. run these commands in separate terminals on a single machine.  First, the `aggregator`, with whom client code directly communicates:

```bash
python scripts/run_aggregator.py --host localhost --port 8082
```

This server receives and accumulates layers and metadata, and when it determines a full time-step (usually, one training iteration) has completed for a model, combines and sends them to the `evaluator`:

```bash
python scripts/run_evaluator.py --host localhost --port 8081
```

This server receives an epoch's-worth of layers at a time, i.e. enough to calculate some value for model `M`'s layer `L` at iteration `I`.  It calculates a scalar value, and sends it along to the `frontend`:

```bash
python scripts/run_frontend.py --host localhost --port 8080 [--database FILE]
```

This is the server that collects and presents results, i.e. you can browse to `localhost:8080`.  Right now, the top-level page lists the models, the second level lists the metrics for a given model, and the third level plots the metric.

## Testing

In another terminal, run one of the tests:

```bash
python scripts/run_examples.py --host localhost --port 8082 --epochs 5
```

You should see output on each of the server terminals as the example models train and pass information along.  After the script returns, you can browse to the interface to see (currently, very boring) plots.

## Using in your code

Minimally, it takes two additional lines of code to use *vivisect* for an existing model:

```python
from vivisect.pytorch import probe

model = <DEFINE YOUR MODEL LIKE NORMAL>

probe(model, "localhost", 8080)

<TRAIN YOUR MODEL LIKE NORMAL>
```

This will walk through your model and attach monitors to the forward calls of every operation that ship off their output to the `aggregator` server every time they're invoked.

```python
probe(model, "localhost", 8080, lambda l : True, lambda m, o, id, od : True)
```

## MXNet from source

MXNet is surprisingly easy to install from source, so until the official version gets bumped to include the latest Gluon refinements, you can follow the Build From Source option in [these instructions](https://mxnet.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU).
