
    # TropD Outgoing Longwave Radiation (OLR) metric
# Written by Ori Adam Mar.21.2017
# Methods:
#  '250W'{Default}: the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses 250W/m^2
#  '20W': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses the tropical OLR max minus 20W/m^2
#  'cutoff': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses a specified cutoff value
#  '10Perc': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR is 10# smaller than the tropical OLR maximum
#  'max': the latitude of maximum of tropical olr in each hemisphere with the smoothing paramerer n=6 in TropD_Calculate_MaxLat
#  'peak': the latitude of maximum of tropical olr in each hemisphere with the smoothing parameter n=30 in TropD_Calculate_MaxLat
# Syntax:
# >> [PhiSH PhiNH] = TropD_Metric_OLR(olr,lat,method,Cutoff)
# Input:
# olr(lat) = zonal mean TOA olr (assumed positive)
# lat = equally spaced latitude column vector
# method (optional) = '250W' {default} | 'cutoff' | '10Perc' | '20W' | 'max' | 'peak'
# Cutoff = scalar (optional). For the method 'cutoff', Cutoff specifies the OLR cutoff value. For the 'max' method, Cutoff (optional, default=6) specifies the smoothing parameter n in TropD_Calculate_MaxLat
# Output:
# PhiSH = latitude of near equator OLR threshold crossing in the SH
# PhiNH = latitude of near equator OLR threshold crossing in the NH
    
from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
from TropD_Calculate_ZeroCrossing import TropD_Calculate_ZeroCrossing	

def TropD_Metric_OLR(olr, lat, method='250W', Cutoff='DEFAULT'):

  if Cutoff == 'DEFAULT':
    Cutoff = 250
  else:
    Cutoff_is_set = 1

  try:
    assert (not hasattr(Cutoff, "__len__") and Cutoff >= 1)  
  except AssertionError:
    print 'TropD_Metric_OLR: ERROR : the Cutoff must be >= 1'
  
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
    olr = np.flip(olr,0)
    lat = np.flip(lat,0)
    
  eq_boundary=5
  subpolar_boundary=40
  polar_boundary=60
  # NH
  olr_max_lat_NH = TropD_Calculate_MaxLat(olr[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                    lat[(lat > eq_boundary) & (lat < subpolar_boundary)])
  olr_max_NH = max(olr[(lat > eq_boundary) & (lat < subpolar_boundary)])

  # SH
  olr_max_lat_SH = TropD_Calculate_MaxLat(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
                    lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)])
  olr_max_SH = max(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)])

  if method=='20W':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] - olr_max_NH + 20,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & \
                    (lat > -polar_boundary)],0) - olr_max_SH + 20,\
                    np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))

  elif method=='250W':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] - 250,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0) - 250,\
                    np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))

  elif method=='cutoff':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] - Cutoff,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0) - Cutoff,\
                    np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))
  
  elif method=='10Perc':
    PhiNH = TropD_Calculate_ZeroCrossing(olr[(lat > olr_max_lat_NH) & (lat < polar_boundary)] / olr_max_NH - 0.9,\
                    lat[(lat > olr_max_lat_NH) & (lat < polar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(olr[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0) \
                    / olr_max_SH - 0.9, np.flip(lat[(lat < olr_max_lat_SH) & (lat > -polar_boundary)],0))

  elif method=='max':
    if Cutoff_is_set:
      PhiNH = TropD_Calculate_MaxLat(olr[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                    lat[(lat > eq_boundary) & (lat < subpolar_boundary)], Cutoff)
      PhiSH = TropD_Calculate_MaxLat(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
                    lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)], Cutoff)
    else:
      PhiNH = np.copy(olr_max_lat_NH)
      PhiSH = np.copy(olr_max_lat_SH)
 
  elif method=='peak':
    PhiNH = TropD_Calculate_MaxLat(olr[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                    lat[(lat > eq_boundary) & (lat < subpolar_boundary)],30)
    PhiSH = TropD_Calculate_MaxLat(olr[(lat > -subpolar_boundary) & (lat < -eq_boundary)],\
                    lat[(lat > -subpolar_boundary) & (lat < -eq_boundary)],30)

  else:
    print 'TropD_Metric_OLR: unrecognized method ', method

    PhiNH = np.empty(0)
    PhiSH = np.empty(0)
  
  return PhiSH, PhiNH
    
