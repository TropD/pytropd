============
Installation
============

Packages
============

You'll need to install the following packages:

* `Python 2.7 <http://python.org/download/>`_
* `Numpy and Scipy <https://www.scipy.org/install.html>`_
* `Matplotlib <https://matplotlib.org/users/installing.html>`_ 
* `PyTropD <https://github.com/tropd/pytropd/>`_  

Pip install
===========

  ``pip install pytropd``

Note that this method will only pull the package but not the additional Netcdf validation files needed if you want to run the tutorial and example calculations. You can download the data files from Github below.

From Github
===========

1) Download the tarball from `here <https://pypi.org/project/pytropd/#files>`_

2) Untar the files 

  ``tar -xzvf pytropd-<version>.tar.gz``
  
  ``cd pytropd-<version>``

3) Compile and install the source
    
  ``cd pytropd``

  ``python setup build``

  ``sudo python setup.py install``

From source
===========

1) Clone the repository using

  ``git clone git@github.com:tropd/pytropd.git``

2) Compile and install the source
    
  ``cd pytropd``

  ``python setup build``

  ``sudo python setup.py install``



