from __future__ import division
import numpy as np

def TropD_Calculate_Mon2Season(Fm, Season=np.arange(12), m=0):
  ''' Calculate seasonal means from monthly time series
  Written by Ori Adam Mar.17.2017 as part of TropD package
  Converted to python by Alison Ming Jul.4.2017

  Syntax:
  >> F = TropD_Calculate_Mon2Season(Fm,Season,m)

  Positional arguments:
  Fm -- array of dimensions (time, latitude, level) or (time, level) or (time, latitude) 
  Season -- vector of months e.g., [0,1,11] for DJF
  m -- index of first of January

  Keyword arguments:
  m (optional, default = 1) -- index of first of January 
  Season (optional, default = np.arange(12)) -- vector of months e.g., [-1,0,1] for DJF

  Output:
  F = the annual time series of the seasonal means'''

    
  try:
    assert(np.max(Season)<12 and np.min(Season)>=0)
  except AssertionError:
        print 'Season can only include indices from 1 to 12'
  
  End_Index = np.shape(Fm)[0]-m+1 - np.mod(np.shape(Fm)[0]-m+1,12)  
  Fm = Fm[m:End_Index,...]
  F = Fm[m + Season[0]::12,...]
  if len(Season) > 1:
    for s in Season[1:]:
      F = F + Fm[m + s::12,...]
    F = F/len(Season)  

  return F


    
