#!/usr/bin/env python

from setuptools import setup

setup(name="Vivisect",
      version="1.0.0",
      description="",
      author="Tom Lippincott",
      author_email="tom@cs.jhu.edu",
      url="http://hltcoe.jhu.edu",
      maintainer="Tom Lippincott",
      maintainer_email="tom@cs.jhu.edu",
      packages=["vivisect.server",
                "vivisect.mxnet",
                "vivisect.pytorch",
                "vivisect.tensorflow"                
      ],
      package_dir={"vivisect.server" : "src/server",
                   "vivisect.mxnet" : "src/mxnet",
                   "vivisect.pytorch" : "src/pytorch",
                   "vivisect.tensorflow" : "src/tensorflow"                   
      },
      scripts=[],
      install_requires=[]
     )
