from __future__ import division
import numpy as np
import scipy as sp
from scipy import integrate
from functions import *
from metrics import *

def find_nearest(array, value):
  ''' Find the index of the item in the array nearest to the value 
  '''
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

def TropD_Calculate_MaxLat(F,lat,n=int(6)):
  ''' Find latitude of absolute maximum value for a given interval

  Converted to python by Paul Staten Jul.29.2017

  Positional arguments:
  F -- vector
  lat -- ordinate vector, the same length as F

  Keyword arguments:
  n (optional, default = 6) -- rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

  Output:
  Ymax -- location of max value of F along lat'''

  try:
    assert(isinstance(n, int)) 
  except AssertionError:
    print 'TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be an integer'
  
  try:
    assert(n>=1) 
  except AssertionError:
    print 'TropD_Calculate_MaxLat: ERROR: the smoothing parameter n must be >= 1'

  try: 
    assert(np.isfinite(F).all())
  except AssertionError:
    print 'TropD_Calculate_MaxLat: ERROR: input field F has NaN values'


  F = F - np.min(F)
  F = F / np.max(F) 

  Ymax = np.trapz((F**n)*lat, lat) / np.trapz(F ** n, lat)

  return Ymax

def TropD_Calculate_Mon2Season(Fm, Season=np.arange(12), m=0):
  ''' Calculate seasonal means from monthly time series
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


    

def TropD_Calculate_StreamFunction(V, lat, lev, *args,**kwargs):
  ''' TropD calculate the streamfunction by integrating the meridional wind from top of the atmosphere to the surface

  Converted to python by Alison Ming Jul.4.2017

  Syntax:
  >> psi = TropD_Calculate_StreamFunction(V,lat,lev)

  Positional arguments:
  V -- zonal-mean meridional wind with dimensions (lat, lev)
  lat -- equally spaced latitude vector
  lev -- vertical level vector in hPa

  Output:
  psi -- the streamfunction psi(lat,lev) '''

    
  EarthRadius = 6371220.0
  EarthGrav = 9.80616
  B = np.ones(np.shape(V)) 
  # B = 0 for subsurface data
  B[np.isnan(V)]=0
  psi = np.zeros(np.shape(V))

  COS = np.repeat(np.cos(lat*np.pi/180), len(lev), axis=0).reshape(len(lat),len(lev))

  psi = (EarthRadius/EarthGrav) * 2 * np.pi \
       * sp.integrate.cumtrapz(B * V * COS, lev*100, axis=1, initial=0) 
  
  return psi

def TropD_Calculate_TropopauseHeight(T ,P, Z=None):
  ''' Calculate the Tropopause Height in isobaric coordinates 

  Converted to python by Alison Ming Jul.4.2017

  Based on the method described in Birner (2010), according to the WMO definition: first level at 
  which the lapse rate <= 2K/km and for which the lapse rate <= 2K/km in all levels at least 2km 
  above the found level 

  Positional arguments:
  T -- Temperature array of dimensions (latitude, levels) on (longitude, latitude, levels)
  P -- pressure levels in hPa

  Keyword arguments:
  Z (optional) -- geopotential height [m] or any field with the same dimensions as T

  Output:
  Pt(lat) or Pt(lon,lat) = tropopause level in hPa 
  Ht(lat) or Ht(lon,lat) = the field Z evaluated at the tropopause. For Z=geopotential heigt, Ht is the tropopause altitude in m '''


  Rd = 287.04
  Cpd = 1005.7
  g = 9.80616
  k = Rd/Cpd
  PI = (np.linspace(1000,1,1000)*100)**k
  Factor = g/Cpd * 1000
  

  if len(np.shape(T)) == 2:
    T = np.expand_dims(T, axis=0)
    Z = np.expand_dims(Z, axis=0)
  # make P monotonically decreasing
  if P[-1] > P[0]:
    P = np.flip(P,0)
    T = np.flip(T,2)
    if Z.any():
      Z = np.flip(Z,2)

  Pk = np.tile((P*100)**k, (np.shape(T)[0], np.shape(T)[1], 1))
  Pk2 = (Pk[:,:,:-1] + Pk[:,:,1:])/2
  
  T2 = (T[:,:,:-1] + T[:,:,1:])/2
  Pk1 = np.squeeze(Pk2[0,0,:])

  Gamma = (T[:,:,1:] - T[:,:,:-1])/(Pk[:,:,1:] - Pk[:,:,:-1]) *\
          Pk2 / T2 * Factor
  Gamma = np.reshape(Gamma, (np.shape(Gamma)[0]*np.shape(Gamma)[1], np.shape(Gamma)[2]))

  T2 = np.reshape(T2, (np.shape(Gamma)[0], np.shape(Gamma)[1]))
  Pt = np.zeros((np.shape(T)[0]*np.shape(T)[1], 1))
  
  for j in range(np.shape(Gamma)[0]):
    G_f = sp.interpolate.interp1d(Pk1, Gamma[j,:], kind='linear', fill_value='extrapolate')
    G1 = G_f(PI)
    T2_f = sp.interpolate.interp1d(Pk1,T2[j,:], kind='linear', fill_value='extrapolate')
    T1 = T2_f(PI)
    idx = np.squeeze(np.where((G1 <=2) & (PI < (550*100)**k) & (PI > (75*100)**k)))
    Pidx = PI[idx] 

    if np.size(Pidx):
      for c in range(len(Pidx)):
        dpk_2km =  -2000 * k * g / Rd / T1[c] * Pidx[c]
        idx2 = find_nearest(Pidx[c:], Pidx[c] + dpk_2km)

        if sum(G1[idx[c]:idx[c]+idx2+1] <= 2)-1 == idx2:
          Pt[j]=Pidx[c]
          break
    else:
      Pt[j] = np.nan
      
 
  Pt = Pt ** (1 / k) / 100
    
  if Z.any():
    Zt =  np.reshape(Z, (np.shape(Z)[0]*np.shape(Z)[1], np.shape(Z)[2]))
    Ht =  np.zeros((np.shape(T)[0]*np.shape(T)[1]))
    
    for j in range(np.shape(Ht)[0]):
      f = sp.interpolate.interp1d(P, Zt[j,:])
      Ht[j] = f(Pt[j])

    Ht = np.reshape(Ht, (np.shape(T)[0], np.shape(T)[1]))
    Pt = np.reshape(Pt, (np.shape(T)[0], np.shape(T)[1]))
    return Pt, Ht
        #disp('TropD_Calculate_TropopauseHeight: ERROR :  T and Z must have the same dimensions')
  
  else:
    
    Pt = np.reshape(Pt, (np.shape(T)[0], np.shape(T)[1]))
    return Pt
from types import *
    
def TropD_Calculate_ZeroCrossing(F, lat, Lat_Uncertainty=0.0,*args,**kwargs):

  '''Find latitude of zero crossing

  Converted to python by Paul Staten Jul.29.2017
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
    
