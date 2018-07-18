
# TropD Tropopause break (TPB) metric
# Written by Ori Adam Mar.17.2017
# Methods:
# 'max_gradient': The latitude of maximal poleward gradient of the tropopause height
# 'cutoff': The most equatorward latitude where the tropopause crosses a prescribed cutoff value
# 'max_potemp': The latitude of maximal difference between the potential temperature at the tropopause and at the surface

    # Syntax:
# >> [PhiSH PhiNH] = TropD_Metric_TPB(T) & (lat,lev,method)
# Or
# >> [PhiSH PhiNH] = TropD_Metric_TPB(T) & (lat,lev,method,Z)
# Or
# >> [PhiSH PhiNH] = TropD_Metric_TPB(T) & (lat,lev,method,Z,Cutoff)
# Input:
# T(lat,lev) = temperature (K)
# lat = equally spaced latitude column vector
# lev = pressure levels column vector in hPa
# method (optional) = 'max_gradient' {default} |  'max_potemp'  |  'cutoff'  |
# Z(lat,lev) (optional) = geopotential height (m)
# Cutoff (optional, scalar) = geopotential height (m) cutoff that marks the location of the tropopause break; Default value = 15000m
# Output:
# PhiSH = latitude of tropopause break in the SH
# PhiNH = latitude of tropopause break in the NH


from __future__ import division
import numpy as np
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
from TropD_Calculate_TropopauseHeight import TropD_Calculate_TropopauseHeight 

def TropD_Metric_TPB(T, lat, lev, method='max_gradient', n=0, Z=None, Cutoff=15*1000):

  Rd = 287.04
  Cpd = 1005.7
  k = Rd / Cpd
  try:
    assert (not hasattr(n, "__len__") and n >= 0)  
  except AssertionError:
    print 'TropD_Metric_TPB: ERROR : the smoothing parameter n must be >= 0'

  polar_boundary=70

  if method=='max_gradient':
    Pt = TropD_Calculate_TropopauseHeight(T,lev)
    Ptd = np.diff(Pt) / (lat[1] - lat[0])
    lat2 = (lat[1:] + lat[:-1]) / 2

    if (n >= 1):
      PhiNH = TropD_Calculate_MaxLat(Ptd[:,(lat2 > 0) & (lat2 < polar_boundary)],\
              lat2[(lat2 > 0) & (lat2 < polar_boundary)], n)
      PhiSH = TropD_Calculate_MaxLat(-Ptd[:,(lat2 > -polar_boundary) & (lat2 < 0)],\
              lat2[(lat2 > -polar_boundary) & (lat2 < 0)], n)
    
    else:
      PhiNH = TropD_Calculate_MaxLat(Ptd[:,(lat2 > 0) & (lat2 < polar_boundary)],\
              lat2[(lat2 > 0) & (lat2 < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(-Ptd[:,(lat2 > -polar_boundary) & (lat2 < 0)],\
              lat2[(lat2 > -polar_boundary) & (lat2 < 0)])
    
  elif method=='max_potemp':
    XF = np.tile((lev / 1000) ** k, (len(lat), 1))
    PT = T / XF
    Pt, PTt = TropD_Calculate_TropopauseHeight(T, lev, Z=PT)
    PTdif = PTt - np.nanmin(PT, axis = 1)
    
    if (n >= 1):
      PhiNH = TropD_Calculate_MaxLat(PTdif[:,(lat > 0) & (lat < polar_boundary)],\
              lat[(lat > 0) & (lat < polar_boundary)], n)
      PhiSH = TropD_Calculate_MaxLat(PTdif[:,(lat > - polar_boundary) & (lat < 0)],\
              lat[(lat > -polar_boundary) & (lat < 0)], n)
    
    else:
      PhiNH = TropD_Calculate_MaxLat(PTdif[:,(lat > 0) & (lat < polar_boundary)],\
              lat[(lat > 0) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_MaxLat(PTdif[:,(lat > - polar_boundary) & (lat < 0)],\
              lat[(lat > -polar_boundary) & (lat < 0)])
   
  elif method=='cutoff':
    Pt, Ht = TropD_Calculate_TropopauseHeight(T, lev, Z)
    
    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
      Ht = np.flip(Ht)
      lat = np.flip(lat)
      polar_boundary = 60
      
      PhiNH = TropD_Calculate_ZeroCrossing(Ht[(lat > 0) & (lat < polar_boundary)] - Cutoff,
              lat[(lat > 0) & (lat < polar_boundary)])
      PhiSH = TropD_Calculate_ZeroCrossing(np.flip(Ht[(lat < 0) & (lat > -polar_boundary)]) - Cutoff,
              np.flip(lat[(lat < 0) & (lat > -polar_boundary)], 0))
  
  else:
    print 'TropD_Metric_TPB: ERROR : Unrecognized method ', method

  return PhiSH, PhiNH
  
