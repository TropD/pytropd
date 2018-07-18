
    # TropD near-surface zonal wind metric
# Written by Ori Adam Mar.21.2017
# Methods:
#  'zero_crossing': the first subtropical latitude where near-surface zonal wind changes from negative to positive
# Syntax:
# >> [PhiSH PhiNH] = TropD_Metric_UAS(U,lat,lev)
# or
# >> [PhiSH PhiNH] = TropD_Metric_UAS(U,lat,lev,method,Lat_Uncertainty)
# Input:
# U(lat,lev) = zonal mean zonal wind
# or 
# U(lat,1) = zonal mean surface wind
# lat = equally spaced latitude column vector
# lev = vertical level vector in hPa units 
# (set lev=1 for single-level input zonal wind U(lat,1))
# method (optional) = 'zero_crossing' {default}
# Lat_Uncertainty (optional) = [Degrees latitude, default = 0] the minimal distance allowed between the first and second zero crossings
# Output:
# PhiSH = latitude of first subtropical zero crossing in the SH
# PhiNH = latitude of first subtropical zero crossing in the NH

from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
from TropD_Calculate_ZeroCrossing import TropD_Calculate_ZeroCrossing	

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def TropD_Metric_UAS(U, lat, lev=np.array([1]), method='zero_crossing', Lat_Uncertainty = 0):
  try:
    assert (Lat_Uncertainty >= 0)  
  except AssertionError:
    print 'TropD_Metric_PSI: ERROR : Lat_Uncertainty must be >= 0'
    
    
  if len(lev) > 1:
    uas = U[:,find_nearest(lev, 850)]
  else:
    uas = np.copy(U)
    
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      uas = np.flip(uas)
      lat = np.flip(lat)

  # define latitudes of boundaries certain regions 
  eq_boundary=5
  subpolar_boundary=30
  polar_boundary=60

  # NH
  uas_min_lat_NH = TropD_Calculate_MaxLat(-uas[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                   lat[(lat > eq_boundary) & (lat < subpolar_boundary)])
  # SH
  uas_min_lat_SH = TropD_Calculate_MaxLat(-uas[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
      lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)])
  try:
    assert(method=='zero_crossing')
    PhiNH = TropD_Calculate_ZeroCrossing(uas[(lat > uas_min_lat_NH) & (lat < polar_boundary)],\
            lat[(lat > uas_min_lat_NH) & (lat < polar_boundary)], Lat_Uncertainty)
    # flip arrays to find the most equatorward zero crossing
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(uas[(lat < uas_min_lat_SH) & (lat > -polar_boundary)],0),\
            np.flip(lat[(lat < uas_min_lat_SH) & (lat > -polar_boundary)],0), Lat_Uncertainty)
    return PhiNH, PhiSH
  except AssertionError:
    print 'TropD_Metric_UAS: ERROR : unrecognized method ', method

  

