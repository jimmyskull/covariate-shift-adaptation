#!/usr/bin/env python

from setuptools import setup

NUMPY_MIN_VERSION = '1.8.2'
TORCH_MIN_VERSION = '0.4.0'
SKLEARN_MIN_VERSION = '0.19.0'

setup(
    name='pantaray',
    version='1.0',
    description='Covariate Shift Adaption algorithms',
    license='BSD 3-Clause License',
    author='Scott Brownlie and Paulo Roberto Urio',
    author_email='',
    url='https://github.com/ayeright/covariate-shift-adaptation',
    packages=['pantaray'],
    install_requires=[
        f'numpy>={NUMPY_MIN_VERSION}',
        f'torch>={TORCH_MIN_VERSION}',
        f'scikit-learn>={SKLEARN_MIN_VERSION}',
    ],
)
