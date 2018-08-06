from __future__ import division
import numpy as np
import scipy as sp
from scipy import integrate
from TropD_Calculate_MaxLat import TropD_Calculate_MaxLat
from TropD_Calculate_ZeroCrossing import TropD_Calculate_ZeroCrossing	
from TropD_Calculate_StreamFunction import TropD_Calculate_StreamFunction 
    
def find_nearest(array, value):
  ''' Find the index of the item in the array nearest to the value 
  '''
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

def TropD_Metric_PSI(V, lat, lev, method='Psi_500', Lat_Uncertainty=0):
  '''TropD PSI metric 
  Latitude of the meridional mass streamfunction subtropical zero crossing
  Written by Ori Adam Mar.20.2017
  Edited by Alison Ming Jul.4.2017
     
  Positional arguments:
  V(lat,lev) -- zonal-mean meridional wind
  lat -- latitude vector
  lev -- vertical level vector in hPa units

  Keyword arguments:  
  method (optional) -- 'Psi_500' (default) |  'Psi_500_10Perc'  |  'Psi_300_700' |  'Psi_500_Int'  |  'Psi_Int'
  
  'Psi_500'{default}: Zero crossing of the stream function (Psi) at the 500hPa level
  'Psi_500_10Perc': Crossing of 10# of the extremum value of Psi in each hemisphre at the 500hPa level
  'Psi_300_700': Zero crossing of Psi vertically averaged between the 300hPa and 700 hPa levels
  'Psi_500_Int': Zero crossing of the vertically-integrated Psi at the 500 hPa level
  'Psi_Int'    : Zero crossing of the column-averaged Psi
    
  Lat_Uncertainty (optional) -- The minimal distance allowed between the first and second zero crossings. For example, for Lat_Uncertainty = 10, the function will return a NaN value if a second zero crossings is found within 10 degrees of the most equatorward zero crossing.   
  
  Outputs:
  PhiSH -- latitude of Psi zero crossing in the SH
  PhiNH -- latitude of Psi zero crossing in the NH
  '''

  try:
    assert (Lat_Uncertainty >= 0)  
  except AssertionError:
    print 'TropD_Metric_PSI: ERROR : Lat_Uncertainty must be >= 0'
    
  subpolar_boundary=30
  polar_boundary=60
    
  Psi = TropD_Calculate_StreamFunction(V, lat, lev)
  COS = np.repeat(np.cos(lat*np.pi/180), len(lev), axis=0).reshape(len(lat),len(lev))
  
  # make latitude vector monotonically increasing
  if lat[-1] < lat[0]:
      Psi = np.flip(Psi, 0)
      lat = np.flip(lat, 0)
    
  Psi[np.isnan(Psi)]=0
    
  if ( method=='Psi_500' or method=='Psi_500_10Perc'):
    # Use Psi at the level nearest to 500 hPa
    P = Psi[:,find_nearest(lev, 500)]
  elif method == 'Psi_300_700':
    # Use Psi averaged between the 300 and 700 hPa level
    P = np.trapz(Psi[:,(lev <= 700) & (lev >= 300)] * COS[:,(lev <= 700) & (lev >= 300)],\
                  lev[(lev <= 700) & (lev >= 300)]*100, axis=1)

  elif method == 'Psi_500_Int':
    # Use integrated Psi from p=0 to level mearest to 500 hPa
    PPsi = sp.integrate.cumtrapz(Psi*COS, lev, axis=1)
    P = PPsi[:,find_nearest(lev, 500)]
  
  elif method == 'Psi_Int':
    # Use vertical mean of Psi 
    P = np.trapz(Psi*COS, lev, axis=1)
  
  else:
    print 'TropD_Metric_PSI: ERROR : Unrecognized method ', method
  
    
  # 1. Find latitude of maximal (minimal) tropical Psi in the NH (SH)
  # 2. Find latitude of minimal (maximal) subtropical Psi in the NH (SH)
  # 3. Find the zero crossing between the above latitudes

  # NH
  Lmax = TropD_Calculate_MaxLat(P[(lat > 0) & (lat < subpolar_boundary)],\
                                lat[(lat > 0) & (lat < subpolar_boundary)])

  Lmin = TropD_Calculate_MaxLat(-P[(lat > Lmax) & (lat < polar_boundary)],\
                                lat[(lat > Lmax) & (lat < polar_boundary)])
  if method=='Psi_500_10Perc':
    Pmax = max(P[(lat > 0) & (lat < subpolar_boundary)])
    PhiNH = TropD_Calculate_ZeroCrossing(P[(lat > Lmax) & (lat < Lmin)] - 0.1*Pmax,\
            lat[(lat > Lmax) & (lat < Lmin)])

  else:
    PhiNH = TropD_Calculate_ZeroCrossing(P[(lat > Lmax) & (lat < Lmin)],\
            lat[(lat > Lmax) & (lat < Lmin)], Lat_Uncertainty=Lat_Uncertainty)
  
  # SH
  Lmax = TropD_Calculate_MaxLat(-P[(lat < 0) & (lat > -subpolar_boundary)],\
         lat[(lat < 0) & (lat > -subpolar_boundary)])

  Lmin = TropD_Calculate_MaxLat(P[(lat < Lmax) & (lat > -polar_boundary)],\
         lat[(lat < Lmax) & (lat > -polar_boundary)])

  if method=='Psi_500_10Perc':
    Pmin = min(P[(lat < 0) & (lat > -subpolar_boundary)])
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(P[(lat < Lmax) & (lat > Lmin)], 0) + 0.1*Pmin,\
            np.flip(lat[(lat < Lmax) & (lat > Lmin)], 0))
  else:
    PhiSH = TropD_Calculate_ZeroCrossing(np.flip(P[(lat < Lmax) & (lat > Lmin)], 0),\
            np.flip(lat[(lat < Lmax) & (lat > Lmin)], 0), Lat_Uncertainty=Lat_Uncertainty)
  return PhiSH, PhiNH

