# Vivisect: let's slice you open and poke around!

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

## Quick start

Install the library and the supported frameworks:

```python
pip install . --user
pip install pytorch mxnet tensorflow --user
```

*Note: Vivisect requires an MXNet version > 1.2 which includes the ability to attach hooks: until then, you'll need to [install from source](www.mxnet.com) or use PyTorch/Tensorflow*

Start the server in a terminal:

```bash
python -m vivisect.server --host localhost --port 8080
```

In another terminal, run one of the tests:

```bash
python scripts/run_examples.py --host localhost --port 8080
```

You should see some output on the server terminal as the example models train and pass tensors to it.

## Using in your code

Minimally, it takes two additional lines of code to use *vivisect* for an existing model:

```python
from vivisect.pytorch import probe

model = <DEFINE YOUR MODEL LIKE NORMAL>

probe(model, "localhost", 8080)

<TRAIN YOUR MODEL LIKE NORMAL>
```

## MXNet from source

MXNet is surprisingly easy to install from source, so until the official version gets bumped to include 
