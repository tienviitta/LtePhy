#!/usr/bin/env python

from distutils.core import setup

setup(
    name='LtePhy',
    version='1.0',
    description='LTE Physical Layer Visualizer and Simulator',
    author='Petri J. Väisänen',
    author_email='petri.j.vaisanen@gmail.com',
    url='https://www.tienviitta.com/ltephy/',
    install_requires=['pytest'],
    packages=['ltephy', 'enb', 'ue'],
)
