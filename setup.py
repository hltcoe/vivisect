#!/usr/bin/env python

import os.path
from glob import glob
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
      packages=["vivisect",
                "vivisect.servers",
                "vivisect.mxnet",
                "vivisect.gluon",
                "vivisect.pytorch",
                "vivisect.tensorflow"
      ],
      package_dir={"vivisect" : "src",
                   "vivisect.servers" : "src/servers",
                   "vivisect.mxnet" : "src/mxnet",
                   "vivisect.gluon" : "src/gluon",
                   "vivisect.pytorch" : "src/pytorch",
                   "vivisect.tensorflow" : "src/tensorflow",
      },
      package_data={"vivisect" : ["logo.png"]},
      include_package_data=True,
      dependency_links=["git+https://github.com/OpenNMT/OpenNMT-py@master#egg=OpenNMT-py-0.2",
                        "git+https://github.com/pytorch/text@master#egg=torchtext-0.2.4"],
      install_requires=["flask>=1.0.0", 
                        "plotly>=2.7.0", 
                        "mxnet-mkl>1.2.0", 
                        "torch==0.4.0", 
                        "tensorflow>=1.8.0", 
                        "sockeye>=1.18.28",
                        "future==0.16.0",
                        "torchtext==0.2.4",
                        "scikit-learn==0.19.1",
                        "scipy==1.1.0",
                        "gluonnlp==0.3.3",
                        "OpenNMT-py==0.2"],
      scripts=glob(os.path.join("scripts", "*")),
     )
