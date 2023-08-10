#!/usr/bin/env python

# PyTropD installation script
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="pytropd",
    version="2.12",
    description="Calculation of metrics of tropical width",
    long_description=long_description,
    license="GPL-3",
    author="Alison Ming, Paul William Staten, Samuel Smith",
    author_email="admg26@gmail.com",
    url="https://tropd.github.io/pytropd/index.html",
    requires=["numpy", "matplotlib", "scipy"],
    install_requires=["numpy>=1.19", "scipy>=1.5"],
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
    package_data={
        "pytropd/ValidationData": ["*.nc"],
        "pytropd/ValidationMetrics": ["*.nc"],
    },
)
