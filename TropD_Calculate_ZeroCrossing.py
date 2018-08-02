from __future__ import division
from types import *
import numpy as np
    
def TropD_Calculate_ZeroCrossing(F, lat, Lat_Uncertainty=0.0,*args,**kwargs):

  '''Find latitude of zero crossing

  Written by Ori Adam Mar.17.2017 as part of TropD package
  Edited by Ori Adam Jun.12.2017
  Converted to python by Alison Ming Jul.5.2017
  Find the first (with increasing index) zero crossing of the function F
  in the interval lat_int

  Syntax:
  >>ZC = Calculate_ZeroCrossing(F,lat,Lat_Uncertainty=Lat_Uncertainty)

  Positional arguments:
  F -- vector
  lat -- latitude vector (same length as F)

  Keyword arguments:
  Lat_Uncertainty (optional) -- [Degrees latitude, default = 0] the minimal
    distance allowed between adjacent zero crossings of indetical sign change
    for example, for Lat_Uncertainty = 10, if the most equatorward zero crossing
    is from positive to negative, the function will return a Nan value if an
    additional zero crossings from positive to netagive is found within 10 degrees
    of that zero crossing.

  Output:
  ZC -- latitude of zero crossing by linear interpolation'''
  # Make sure a zero crossing exists
  a = np.where(F > 0)[0]
  if len(a) == len(F) or not any(a):
    return np.nan

  # Find first zero crossing in index units.
  D = np.diff(np.sign(F))

  # If more than one zero crossing exists in proximity to the first zero crossing.
  a = np.where(np.abs(D)>0)[0]
  if len(a)>2 and np.abs(lat[a[2]] - lat[a[0]]) < Lat_Uncertainty:
    return np.nan

  a1 = np.argmax(np.abs(D) > 0)
  # if there is an exact zero, use its latitude...
  if np.abs(D[a1])==1:
    ZC = lat[a1]
  else:
    ZC = lat[a1] - F[a1]*(lat[a1+1]-lat[a1])/(F[a1+1]-F[a1])
  return ZC
    
