    
from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat


def find_nearest(array, value):
  ''' Find the index of the item in the array nearest to the value 
  '''
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx
    
def TropD_Metric_EDJ(U, lat, lev=np.array([1]), method='max', n=0):
  
  '''TropD EDJ metric
  
  Latitude of the eddy driven jet (EDJ) 
  Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
  Written by Ori Adam Mar.20.2017
  Edited by Alison Ming Jul.4.2017
  
  Positional arguments:
  U(lat,lev) or U (lat,)-- Zonal mean zonal wind. Also takes surface wind 
  lat -- latitude vector
  lev -- vertical level vector in hPa units

  Keyword arguments:
  method (optional) -- 'max' (default) |  'peak'
  n (optional, default = 6) -- rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

  Outputs:
  PhiSH -- latitude of EDJ in the SH
  PhiNH -- latitude of EDJ in the NH
  '''

  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print 'TropD_Metric_EDJ: ERROR : the smoothing parameter n must be >= 0'
    
  eq_boundary = 15
  polar_boundary = 60
  
  if len(lev) > 1:
    u = U[:,find_nearest(lev, 850)]
  else:
    u = np.copy(U)
    
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

  return PhiSH, PhiNH



