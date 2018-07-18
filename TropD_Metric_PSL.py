
    # TropD PSL metric
# Latitude of maximum of the subtropical sea-level pressure
# Written by Ori Adam Mar.17.2017
# Methods:
# the latitude of maximum of sea-level pressure
    
    # Syntax:
# >> [PhiSH PhiNH] = TropD_Metric_PSL(ps,lat,method)
# Input:
# ps(lat) = sea-level pressure
# lat = equally spaced latitude column vector
# method (optional) = | 'max' {default} | 'peak'
# Output:
# PhiSH = latitude of subtropical sea-level pressure maximum in the SH
# PhiNH = latitude of subtropical sea-level pressure maximum in the NH
    
from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
    
def TropD_Metric_PSL(ps, lat, method='max'):

  eq_boundary=15
  polar_boundary=60
    
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
  
  return PhiNH,PhiSH
    
