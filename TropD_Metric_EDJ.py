    # TropD EDJ metric
# Latitude of the eddy driven jet (EDJ) 
# Written by Ori Adam Mar.20.2017
# Methods:
# Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
    
    # Syntax:
# >> [PhiSH PhiNH] = TropD_Metric_EDJ(U,lat,lev,method)
# Input:
# U(lat,lev) = zonal mean zonal wind. Surface wind or uni-level zonal wind data can also be used as an input field using U(lat,1) 
# lat = equally spaced latitude column vector
# lev = vertical level vector in hPa units
# method (optional) = 'max' {default} |  'peak'
# Output:
# PhiSH = latitude of EDJ in the SH
# PhiNH = latitude of EDJ in the NH
    
from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
    
def TropD_Metric_EDJ(u, lat, lev=np.array([1]), method='max', n=0):
    
  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print 'TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0'
    
  eq_boundary=15
  polar_boundary=60
  
  if len(lev) > 1:
    uas = u[:,find_nearest(lev, 850)]
  else:
    uas = np.copy(u)
    
  if 'max' == method:
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)],n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)],n)

    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)])
  elif method=='peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)],n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)],n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
              lat[(lat > eq_boundary) & (lat < polar_boundary)],30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
              lat[(lat > -polar_boundary) & (lat < -eq_boundary)],30)
  
  else:
    print 'TropD_Metric_EDJ: ERROR: unrecognized method ',method

  return PhiNH, PhiSH



