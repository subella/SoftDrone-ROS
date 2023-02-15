#!/usr/bin/env python3

import os
from setuptools import setup

setup(
    name='keypointserver',
    description='ZMQ Python3 Keypoint Server',
    version='1.0',
    author='Sam Ubellacker',
    author_email='subella@mit.edu',
    packages=['keypointserver'],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=[],
)
