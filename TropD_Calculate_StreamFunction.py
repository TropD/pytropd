from __future__ import division
import numpy as np
import scipy as sp
from scipy import integrate

def TropD_Calculate_StreamFunction(V, lat, lev, *args,**kwargs):
  ''' TropD calculate the streamfunction by integrating the meridional wind from top of the atmosphere to the surface

  Written by Ori Adam Mar.17.2017 as part of TropD package
  Converted to python by Alison Ming Jul.4.2017

  Syntax:
  >> psi = TropD_Calculate_StreamFunction(V,lat,lev)

  Positional arguments:
  V -- zonal-mean meridional wind with dimensions (lat, lev)
  lat -- equally spaced latitude vector
  lev -- vertical level vector in hPa

  Output:
  psi -- the streamfunction psi(lat,lev) '''

    
  EarthRadius=6371220.0
  EarthGrav=9.80616
  B = np.ones(np.shape(V)) 
  # B = 0 for subsurface data
  B[np.isnan(V)]=0
  psi = np.zeros(np.shape(V))

  COS = np.repeat(np.cos(lat), len(lev), axis=0).reshape(len(lat),len(lev))

  psi[:,1:] = (EarthRadius/EarthGrav) * 2 * np.pi \
       * sp.integrate.cumtrapz(B * V * COS, lev*100, 1) 
  
  return psi
