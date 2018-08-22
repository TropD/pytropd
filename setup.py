#!/usr/bin/env python

# PyTropD installation script
from setuptools import setup
from os import path

setup (name="pytropd",
	version="1.0.0",
        description = "Calculation of metrics of tropical width",
	long_description = """
PyTropD is a software package designed to calculate various metrics of tropical width and is the python equivalent of the Matlab package,TropD""",
        license = "GPL-3",
        author="Alison Ming, Paul William Staten",
        author_email="admg26@gmail.com",
        url="",
	requires=['numpy','matplotlib','scipy'],
	packages=["pytropd"],
        classifiers=["Programming Language :: Python :: 2"],
)

