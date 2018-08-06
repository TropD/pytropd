from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
    
def TropD_Metric_PSL(ps, lat, method='max'):

  '''TropD PSL metric
  Latitude of maximum of the subtropical sea-level pressure
  Written by Ori Adam Mar.17.2017
  Edited by Alison Ming Jul.4.2017
  
  Positional arguments:
  ps(lat,) -- sea-level pressure
  lat -- equally spaced latitude column vector

  Keyword arguments:
  method (optional) -- 'max' (default) | 'peak'
  
  Outputs:
  PhiSH -- latitude of subtropical sea-level pressure maximum in the SH
  PhiNH -- latitude of subtropical sea-level pressure maximum in the NH
  '''

  eq_boundary = 15
  polar_boundary = 60
    
  if 'max' == method:
    PhiNH = TropD_Calculate_MaxLat(ps[(lat > eq_boundary) & (lat < polar_boundary)],\
            lat[(lat > eq_boundary) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_MaxLat(ps[(lat > -polar_boundary) & (lat < -eq_boundary)],\
            lat[(lat > -polar_boundary) & (lat < -eq_boundary)])

  elif 'peak' == method:
    PhiNH = TropD_Calculate_MaxLat(ps[(lat > eq_boundary) & (lat < polar_boundary)],\
            lat[(lat > eq_boundary) & (lat < polar_boundary)], 30)
    PhiSH = TropD_Calculate_MaxLat(ps[(lat > -polar_boundary) & (lat < -eq_boundary)],\
            lat[(lat > -polar_boundary) & (lat < -eq_boundary)], 30)
  else:
    print 'TropD_Metric_PSL: ERROR: unrecognized method ',method
  
  return PhiSH, PhiNH
    
