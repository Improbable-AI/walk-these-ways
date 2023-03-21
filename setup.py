from setuptools import find_packages
from distutils.core import setup

setup(
    name='go1_gym',
    version='1.0.0',
    author='Gabriel Margolis',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='gmargo@mit.edu',
    description='Toolkit for deployment of sim-to-real RL on the Unitree Go1.',
    install_requires=['ml_logger==0.8.117',
                      'ml_dash==0.3.20',
                      'jaynes>=0.9.2',
                      'params-proto==2.10.5',
                      'gym>=0.14.0',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.23.5'
                      ]
)
