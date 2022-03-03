#!/usr/bin/env python
from setuptools import setup, find_packages

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
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=install_requires,
)