#!/usr/bin/env python

from setuptools import setup

frameworks = ["pytorch", "mxnet", "tensorflow"]
model_types = ["mlp", "rnn"]
special_cases = ["sockeye", "opennmt"]

setup(name="Vivisect",
      version="1.0.1",
      description="",
      author="Tom Lippincott",
      author_email="tom@cs.jhu.edu",
      url="http://hltcoe.jhu.edu",
      maintainer="Tom Lippincott",
      maintainer_email="tom@cs.jhu.edu",
      packages=["vivisect.servers",
                "vivisect.mxnet",
                "vivisect.pytorch",
                "vivisect.tensorflow"
      ],
      package_dir={"vivisect.servers" : "src/servers",
                   "vivisect.mxnet" : "src/mxnet",
                   "vivisect.pytorch" : "src/pytorch",
                   "vivisect.tensorflow" : "src/tensorflow",
      },
      install_requires=["flask>=1.0.0", "plotly", "mxnet>1.2.0", "torch", "tensorflow", "sockeye", "OpenNMT-py>=0.2"]
     )
