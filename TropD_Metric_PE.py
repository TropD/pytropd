from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
from TropD_Calculate_ZeroCrossing import TropD_Calculate_ZeroCrossing	
    
def TropD_Metric_PE(pe,lat,method='zero_crossing',Lat_Uncertainty=0.0):

  '''TropD Precipitation minus Evaporation (PE) metric
  Written by Ori Adam Mar.21.2017
  Edited by Alison Ming Jul.4.2017
     
  Positional arguments:
  pe(lat,) -- zonal-mean precipitation minus evaporation
  lat -- equally spaced latitude column vector

  Keyword arguments:
  method -- 'zero_crossing': the first latitude poleward of the subtropical minimum where P-E changes from negative to positive values. Only one method so far.
  Lat_Uncertainty (optional) -- The minimal distance allowed between the first and second zero crossings along lat

  Output:
  PhiSH -- latitude of first subtropical P-E zero crossing in the SH
  PhiNH -- latitude of first subtropical P-E zero crossing in the NH
  '''    
    
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      pe = np.flip(pe)
      lat = np.flip(lat)
    
  # The gradient of PE is used to determine whether PE becomes positive at the zero crossing
  ped = np.interp(lat, (lat[:-1] + lat[1:])/2.0, np.diff(pe))
    
  # define latitudes of boundaries certain regions 
  eq_boundary=5
  subpolar_boundary=50
  polar_boundary=70

  try:
    assert(method=='zero_crossing')
    # NH
    M1 = TropD_Calculate_MaxLat(-pe[(lat > eq_boundary) & (lat < subpolar_boundary)],\
                   lat[(lat > eq_boundary) & (lat < subpolar_boundary)], 30)
    ZC1 = TropD_Calculate_ZeroCrossing(pe[(lat > M1) & (lat < polar_boundary)], \
                   lat[(lat > M1) & (lat < polar_boundary)], Lat_Uncertainty)
    if np.interp(ZC1, lat, ped) > 0:
      PhiNH = ZC1
    else:
      PhiNH = TropD_Calculate_ZeroCrossing(pe[(lat > ZC1) & (lat < polar_boundary)], \
                    lat[(lat > ZC1) & (lat < polar_boundary)], Lat_Uncertainty)
    
    # SH
    # flip arrays to find the most equatorward zero crossing
    M1 = TropD_Calculate_MaxLat(np.flip(-pe[(lat < -eq_boundary) & (lat > -subpolar_boundary)],0),\
                   np.flip(lat[(lat < -eq_boundary) & (lat > -subpolar_boundary)],0), 30)               
    ZC1 = TropD_Calculate_ZeroCrossing(np.flip(pe[(lat < M1) & (lat > -polar_boundary)],0), \
                   np.flip(lat[(lat < M1) & (lat > -polar_boundary)],0), Lat_Uncertainty)

    if np.interp(ZC1, lat, ped) < 0:
      PhiSH = ZC1
    else:
      PhiSH = TropD_Calculate_ZeroCrossing(np.flip(pe[(lat < ZC1) & (lat > -polar_boundary)],0), \
                    np.flip(lat[(lat < ZC1) & (lat > -polar_boundary)],0), Lat_Uncertainty)

    return PhiSH, PhiNH
  except AssertionError:
    print 'TropD_Metric_PE: ERROR : unrecognized method ',method
