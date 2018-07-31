from __future__ import division
import numpy as np
import scipy as sp
from scipy import interpolate 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def TropD_Calculate_TropopauseHeight(T ,P, Z=None):
  ''' Calculate the Tropopause Height in isobaric coordinates 

  Written by Ori Adam Mar.17.2017 as part of TropD package
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
