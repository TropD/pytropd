
# TropD Subtropical Jet (STJ) metric
# Written by Ori Adam Mar.20.2017
# Methods:
#  'adjusted' {Default}: Latitude of maximum (smoothing parameter n=6) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level,
#                        poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude
#  'adjusted_peak': Latitude of maximum (smoothing parameter n=30) of the zonal wind averaged between the 100 and 400 hPa levels minus the zonal mean zonal wind at the level closes to the 850 hPa level,
#                   poleward of 10 degrees and equatorward of the Eddy Driven Jet latitude
#  'core': Latitude of maximum of the zonal wind (smoothing parameter n=6) averaged between the 100 and 400 hPa levels,
#          poleward of 10 degrees and equatorward of 70 degrees
#  'core_peak': Latitude of maximum of the zonal wind (smoothing parameter n=30) averaged between the 100 and 400 hPa levels,
#               poleward of 10 degrees and equatorward of 70 degrees

    # Syntax:
# >> [PhiSH PhiNH] = TropD_Metric_STJ(U,lat,lev,method)
# Input:
# U(lat,lev) = zonal mean zonal wind
# lat = equally spaced latitude column vector
# lev = vertical level vector in hPa units
# method (optional) = 'adjusted' {default}| 'core' | 'adjusted_peak' | 'core_peak'
# Output:
# PhiSH = latitude of STJ in the SH
# PhiNH = latitude of STJ in the NH

from __future__ import division
import numpy as np
import scipy as sp
from scipy import integrate
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
from TropD_Calculate_ZeroCrossing import TropD_Calculate_ZeroCrossing	
from TropD_Calculate_StreamFunction import TropD_Calculate_StreamFunction 
from TropD_Metric_EDJ import TropD_Metric_EDJ

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def TropD_Metric_STJ(U, lat, lev, method='adjusted', n=0):
  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print 'TropD_Metric_STJ: ERROR : the smoothing parameter n must be >= 0'

  eq_boundary=10
  polar_boundary=70

  lev_int = lev[(lev >= 100) & (lev <= 400)]

  if (method=='adjusted' or method=='adjusted_peak'): 
    idx_850 = find_nearest(lev, 850)

    # Pressure weighted vertical mean of U minus near surface U
    if len(lev_int) > 1:
      u = np.trapz(U[:, (lev >= 100) & (lev <= 400)], lev_int, axis=1) \
          / (lev_int[-1] - lev_int[0]) - U[:,idx_850]

    else:
      u = np.mean(U[:,(lev >= 100) & (lev <= 400)], axis=1) - U[:,idx_850]

  elif (method=='core' or method=='core_peak'):
    # Pressure weighted vertical mean of U
    if len(lev_int) > 1:
      u = np.trapz(U[:, (lev >= 100) & (lev <= 400)], lev_int, axis=1) \
          / (lev_int[-1] - lev_int[0])

    else:
      u = np.mean(U[:, (lev >= 100) & (lev <= 400)], axis=1)

  else:
    print 'TropD_Metric_STJ: unrecognized method ',method
    print 'TropD_Metric_STJ: optional methods are: adjusted (default), adjusted_peak, core, core_peak'

  if method=='core':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)], n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)])

  elif method=='core_peak':
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)], n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < - eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < polar_boundary)],\
          lat[(lat > eq_boundary) & (lat < polar_boundary)], 30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > -polar_boundary) & (lat < -eq_boundary)],\
          lat[(lat > -polar_boundary) & (lat < -eq_boundary)], 30)

  elif method=='adjusted':
    PhiSH_EDJ, PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], n)

    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)])
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)])

  elif method=='adjusted_peak':
    PhiSH_EDJ,PhiNH_EDJ = TropD_Metric_EDJ(U,lat,lev)
    if n:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], n)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], n)
    else:
      PhiNH = TropD_Calculate_MaxLat(u[(lat > eq_boundary) & (lat < PhiNH_EDJ)],\
          lat[(lat > eq_boundary) & (lat < PhiNH_EDJ)], 30)
      PhiSH = TropD_Calculate_MaxLat(u[(lat > PhiSH_EDJ) & (lat < -eq_boundary)],\
          lat[(lat > PhiSH_EDJ) & (lat < -eq_boundary)], 30)

  return PhiSH, PhiNH    
