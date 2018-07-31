from __future__ import division
from types import *
import numpy as np

def TropD_Calculate_MaxLat(F,lat,n=int(6)):
  ''' Find latitude of absolute maximum value for a given interval

  Written by Ori Adam Mar.17.2017 as part of TropD package
  Converted to python by Alison Ming Jul.4.2017

  Syntax:
  >> Ymax = TropD_Calculate_MaxLat(F,lat,n=6)

  Positional arguments:
  F -- vector
  lat -- ordinate vector, the same length as F

  Keyword arguments:
  n (optional, default = 6) -- rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

  Output:
  Ymax -- location of max value of F along lat'''

  try:
    assert type(n) is IntType 
    try:
      assert(n>=1) 
    except AssertionError:
        print 'TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be >= 1'
  except AssertionError:
    print 'TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be an integer'

  try: 
    assert(np.isfinite(F).all())
  except AssertionError:
    print 'TropD_Calculate_MaxLat: ERROR: input field F has NaN values'


  F = F - np.min(F)
  F = F / np.max(F) 

  Ymax = np.trapz((F**n)*lat, lat) / np.trapz(F ** n, lat)

  return Ymax
