#!/usr/bin/env python

from setuptools import setup

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
                   "vivisect.tensorflow" : "src/tensorflow"                   
      },
      scripts=[],
      install_requires=["plotly", "mxnet>=1.3.0", "pytorch", "tensorflow", "sockeye"]
     )
