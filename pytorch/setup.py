#!/usr/bin/env python
from setuptools import setup

install_requires = ["webdataset",
                    "pytorch-lightning",
                    "lightning-bolts",
                    "torch_tb_profiler"]

setup(
    name="resnet_sagemaker",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/johnbensnyder/resnet",
    description="Resnet test",
    install_requires=install_requires,
)