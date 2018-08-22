========
Tutorial
========

This tutorial assumes that you have downloaded the code and additional Netcdf validation files needed to run the tutorial and example calculations. These are available on `Github <https://github.com/tropd/pytropd/>`_.

The directory structure is assumed to be:: 

  pytropd
  | -- pytropd               <-- Run the tutorial from inside this directory
  |    | -- tutorial.py
  |    | -- functions.py
  |    | -- __init__.py   
  |    | -- metrics.py
  |    | -- TropD_Example_Calculations.py
  | -- ValidationData
  |    | -- *.nc             <-- Collection of netcdf data files used by 
  |    `                         tutorial.py and TropD_Example_Calculations.py
  ` -- ValidationMetrics
       | -- *.nc             <-- Collection of netcdf validation data files with 
       `                         precomputed metrics used by TropD_Example_Calculations.py

.. currentmodule:: pytropd

First import pytropd and some data.

.. code-block:: python

  In [1]: import pytropd as pyt

  In [2]: from pytropd.tutorial import lat, lev, V 

  In [3]: print V

.. currentmodule:: pytropd

V is a numpy array containing the mean meridional velocity on (lat, levs). We can calculate the metric of tropical width from the mass streamfunction (PSI) as follows:

.. code-block:: python
  
  In [4]: Phi_sh, Phi_nh = pyt.TropD_Metric_PSI(V[j,:,:], lat, lev)

  In [5]: print(Phi_sh, Phi_nh)

More detailed code examples can be found in the file ``TropD_Example_Calculations.py``.
