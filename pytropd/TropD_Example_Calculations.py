from __future__ import division
import os
import numpy as np
import scipy as sp
from scipy.io import netcdf
import pytropd as pyt
import matplotlib.pyplot as plt
from matplotlib import rc

# Example codes for using the TropD package to calculate tropical width metrics
# The code assumes that the current directory is ... tropd/pytropd/
## Set display and meta parameters
y1 = 1979
y2 = 2016
time = np.linspace(y1,y2,12*(y2-y1+1))
#time=linspace(y1,y2 + 1,dot(12,(y2 - y1 + 1)) + 1)
#time=(time(arange(1,end() - 1)) + time(arange(2,end()))) / 2
red_color     = (1,0.3,0.4)
orange_color  = (255/256,140/256,0) 
blue_color    = (0,0.447,0.741)
purple_color  = (0.494,0.184,0.556)
green_color   = (0.466,0.674,0.188)
lightblue_color = (0.301,0.745,0.933)
maroon_color  = (0.635,0.078,0.184)

# used to build absolute path to data files
dirname = os.path.dirname(__file__)

## 1) PSI -- Streamfunction zero crossing
#read meridional velocity V(time,lat,lev), latitude and level
f_V = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/va.nc'),'r')
V = f_V.variables['va'][:]
#Change axes of V to be [time, lat, lev]
V = np.transpose(V, (2,1,0))
lat = f_V.variables['lat'][:]
lev = f_V.variables['lev'][:]
f_V.close()

Phi_psi_nh = np.zeros((np.shape(V)[0],)) # latitude of monthly NH Psi zero crossing
Phi_psi_sh = np.zeros((np.shape(V)[0],)) # latitude of monthly SH Psi zero crossing

for j in range(np.shape(V)[0]):
  # Default method = 'Psi500'
  Phi_psi_sh[j], Phi_psi_nh[j] = pyt.TropD_Metric_PSI(V[j,:,:], lat, lev)


# Calculate metric from annual mean
V_ANN = pyt.TropD_Calculate_Mon2Season(V,season=np.arange(12))

Phi_psi_nh_ANN = np.zeros((np.shape(V_ANN)[0],)) # latitude of NH stream function zero crossing from annual mean V
Phi_psi_sh_ANN = np.zeros((np.shape(V_ANN)[0],)) # latitude of SH stream function zero crossing from annual mean V

for j in range(np.shape(V_ANN)[0]):
  Phi_psi_sh_ANN[j], Phi_psi_nh_ANN[j] = pyt.TropD_Metric_PSI(V_ANN[j,:,:], lat, lev)


plt.figure(1)
plt.subplot(211)
plt.plot(time, Phi_psi_nh, linewidth=1, color=green_color, \
        label=r'Latitude of $\Psi_{500}$ zero crossing from monthly mean V')
plt.plot(np.arange(y1,y2+1)+0.5, Phi_psi_nh_ANN, linewidth=2, color=blue_color,\
        label=r'Latitude of $\Psi_{500}$ zero crossing from annual mean V')
plt.plot(np.arange(y1,y2+1)+0.5, pyt.TropD_Calculate_Mon2Season(Phi_psi_nh, season=np.arange(12)),linewidth=2, color='k',\
        label=r'Latitude of $\Psi_{500}$ zero crossing from annual means of monthly metric values')
plt.xticks(np.arange(1980,2020,5))
plt.ylabel('latitude')
plt.title(r"NH $\Psi_{500}$")
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(time, Phi_psi_sh, linewidth=1, color=green_color)
plt.plot(np.arange(y1,y2+1)+0.5, Phi_psi_sh_ANN, linewidth=2, color=blue_color)
plt.plot(np.arange(y1,y2+1)+0.5, pyt.TropD_Calculate_Mon2Season(Phi_psi_sh, season=np.arange(12)),linewidth=2, color='k')
plt.xticks(np.arange(1980,2020,5))
plt.ylabel('latitude')
plt.title(r"SH $\Psi_{500}$")
plt.show()

# Introduce latitude unertainty condition: no additional zero crossing is allowed within 10 degrees
Phi_psi_nh_L = np.zeros((np.shape(V)[0],)) # latitude of monthly NH Psi zero crossing
Phi_psi_sh_L = np.zeros((np.shape(V)[0],)) # latitude of monthly SH Psi zero crossing

for j in range(np.shape(V)[0]):
  Phi_psi_sh_L[j], Phi_psi_nh_L[j] = pyt.TropD_Metric_PSI(V[j,:,:], lat, lev, method='Psi_500', lat_uncertainty=10)

plt.figure(2)
plt.plot(time,Phi_psi_nh,linewidth=2,color=green_color,\
    label='$\Psi_{500}$ zero crossing from monthly mean V that qualify uncertainty criterion') 
plt.plot(time[np.isnan(Phi_psi_nh_L)], Phi_psi_nh[np.isnan(Phi_psi_nh_L)],marker='*',linestyle='None',markersize=10,color=red_color,\
    label='$\Psi_{500}$ zero crossing from monthly mean V that fail uncertainty criterion')
plt.title(r'NH $\Psi_{500}$')
plt.ylabel('latitude')
plt.legend(loc='best', frameon=False)
plt.xlabel('Year')
plt.show()


## 2) TPB -- Tropopause break latitude
#read temperature T(time,lat,lev), potential height [m] Z(time,lat,lev), latitude and level
f_T = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ta.nc'),'r')
f_Z = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/zg.nc'),'r')
T = f_T.variables['ta'][:]
Z = f_Z.variables['zg'][:]
#Change axes of T and Z to be [time, lat, lev]
T = np.transpose(T, (2,1,0))
Z = np.transpose(Z, (2,1,0))

lat = f_T.variables['lat'][:]
lev = f_T.variables['lev'][:]
f_T.close()
f_Z.close()


Phi_tpb_nh = np.zeros((np.shape(T)[0],)) # latitude of monthly NH tropopause break
Phi_tpb_sh = np.zeros((np.shape(T)[0],)) # latitude of monthly SH tropopause break

for j in range(np.shape(T)[0]):
  ''' Default method = 'max_gradient'. Latitude of maximal difference between 
  potential temperature at the tropopause level and the potential temperature at the surface
  '''
  Phi_tpb_sh[j], Phi_tpb_nh[j] = pyt.TropD_Metric_TPB(T[j,:,:], lat, lev)

# Calculate tropopause break from annual mean
T_ANN = pyt.TropD_Calculate_Mon2Season(T, season=np.arange(12))
Z_ANN = pyt.TropD_Calculate_Mon2Season(Z, season=np.arange(12))

Phi_tpb_nh_ANN = np.zeros((np.shape(T_ANN)[0],))  # latitude of NH tropopause break from annual mean T
Phi_tpb_sh_ANN = np.zeros((np.shape(T_ANN)[0],))  # latitude of SH tropopause break from annual mean T
Phi_tpbZ_nh_ANN = np.zeros((np.shape(T_ANN)[0],)) # latitude of NH tropopause break from annual mean T
Phi_tpbZ_sh_ANN = np.zeros((np.shape(T_ANN)[0],)) # latitude of SH tropopause break from annual mean T
Phi_tpbT_nh_ANN = np.zeros((np.shape(T_ANN)[0],)) # latitude of NH tropopause break from annual mean T
Phi_tpbT_sh_ANN = np.zeros((np.shape(T_ANN)[0],)) # latitude of SH tropopause break from annual mean T

for j in range(np.shape(T_ANN)[0]):
  # (Default) latitude of maximal poleward gradient
  Phi_tpb_sh_ANN[j], Phi_tpb_nh_ANN[j] = pyt.TropD_Metric_TPB(T_ANN[j,:,:], lat, lev, method='max_gradient')
  # latitude of maximal difference in potential temperature between the tropopase and surface 
  Phi_tpbT_sh_ANN[j], Phi_tpbT_nh_ANN[j] = pyt.TropD_Metric_TPB(T_ANN[j,:,:], lat, lev, method='max_potemp')
  # CutoffHeight = 15km marks the height of the tropopause break
  Phi_tpbZ_sh_ANN[j], Phi_tpbZ_nh_ANN[j] = pyt.TropD_Metric_TPB(T_ANN[j,:,:], lat, lev, method='cutoff',\
                                                           Z=Z_ANN[j,:,:,], Cutoff=15*1000)

plt.figure(3)
plt.subplot(211)
plt.plot(time,Phi_tpb_nh,linewidth=1,color=green_color, \
    label='Latitude of tropopause break from monthly mean T -- potential temperature difference')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_tpb_nh_ANN,linewidth=2,color=blue_color,
    label='Latitude of tropopause break from annual mean T -- maximal gradient')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_tpbZ_nh_ANN,linestyle='--',linewidth=1,color=blue_color,\
    label='Latitude of tropopause break from annual mean T -- 15km cutoff height')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_tpbT_nh_ANN,linestyle='--',linewidth=1,color=red_color,\
    label='Latitude of tropopause break from annual mean T -- potential temperature difference')
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_tpb_nh, season=np.arange(12)),color='k',linewidth=2,\
    label='Latitude of tropopause break from annual mean of monthly metric values -- potential temperature difference')
plt.title(r'NH tropopause break')
plt.ylabel('latitude')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(time,Phi_tpb_sh,linewidth=1,color=green_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_tpb_sh_ANN,linewidth=2,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_tpbZ_sh_ANN,linestyle='--',linewidth=1,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_tpbT_sh_ANN,linestyle='--',linewidth=1,color=red_color)
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_tpb_sh, season=np.arange(12)),color='k',linewidth=2)
plt.xlabel('Year')
plt.title(r'SH tropopause break')
plt.ylabel('latitude')
plt.show()

##3) OLR -- OLR cutoff
#Note: OLR is assumed to be positive upwards and in units of W/m^2
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
# read zonal mean monthly clear-sky TOA outgoing longwave radiation olrcs(time,lat)
f_olr = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/rlnt.nc'),'r')
f_olrcs = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/rlntcs.nc'),'r')
olr = -f_olr.variables['rlnt'][:]
olrcs = -f_olrcs.variables['rlntcs'][:]
lat = f_olr.variables['lat'][:]
f_olr.close()
f_olrcs.close()

#Change axes of olr and olrcs to be [time, lat]
olr = np.transpose(olr, (1,0))
olrcs = np.transpose(olrcs, (1,0))

#Calculate annual mean field
olr_ANN = pyt.TropD_Calculate_Mon2Season(olr, season=np.arange(12))
olrcs_ANN = pyt.TropD_Calculate_Mon2Season(olrcs, season=np.arange(12))

Phi_olr_nh = np.zeros((np.shape(olr)[0],))            # latitude of NH olr metric                                    
Phi_olr_sh = np.zeros((np.shape(olr)[0],))            # latitude of SH olr metric

Phi_olr_nh_ANN = np.zeros((np.shape(olr_ANN)[0],))    # latitude of NH olr metric from annual mean olr
Phi_olr_sh_ANN = np.zeros((np.shape(olr_ANN)[0],))    # latitude of SH olr metric from annual mean olr

Phi_olrcs_nh = np.zeros((np.shape(olr)[0],))          # latitude of NH olr metric
Phi_olrcs_sh = np.zeros((np.shape(olr)[0],))          # latitude of SH olr metric

Phi_olrcs_nh_ANN = np.zeros((np.shape(olr_ANN)[0],))  # latitude of NH olr metric from annual mean olr
Phi_olrcs_sh_ANN = np.zeros((np.shape(olr_ANN)[0],))  # latitude of SH olr metric from annual mean olr

Phi_olr20_nh_ANN = np.zeros((np.shape(olr_ANN)[0],))  # latitude of NH olr metric from annual mean olr-20W/m^2 cutoff
Phi_olr20_sh_ANN = np.zeros((np.shape(olr_ANN)[0],))  # latitude of SH olr metric from annual mean olr-20W/m^2 cutoff
Phi_olr240_nh_ANN = np.zeros((np.shape(olr_ANN)[0],)) # latitude of NH olr metric from annual mean olr 240W/m^2 cutoff
Phi_olr240_sh_ANN = np.zeros((np.shape(olr_ANN)[0],)) # latitude of SH olr metric from annual mean olr 240W/m^2 cutoff

for j in range(np.shape(olr)[0]):
  # Default method = '250W'
  Phi_olr_sh[j], Phi_olr_nh[j] = pyt.TropD_Metric_OLR(olr[j,:], lat)
  Phi_olrcs_sh[j], Phi_olrcs_nh[j] = pyt.TropD_Metric_OLR(olrcs[j,:], lat)

for j in range(np.shape(olr_ANN)[0]):
  Phi_olr_sh_ANN[j], Phi_olr_nh_ANN[j] = pyt.TropD_Metric_OLR(olr_ANN[j,:], lat)
  Phi_olrcs_sh_ANN[j], Phi_olrcs_nh_ANN[j] = pyt.TropD_Metric_OLR(olrcs_ANN[j,:], lat)
  Phi_olr20_sh_ANN[j], Phi_olr20_nh_ANN[j] = pyt.TropD_Metric_OLR(olr_ANN[j,:], lat, method='20W')
  Phi_olr240_sh_ANN[j], Phi_olr240_nh_ANN[j] = pyt.TropD_Metric_OLR(olr_ANN[j,:],lat,method='cutoff',Cutoff=240)



plt.figure(4)
plt.subplot(211)
plt.plot(time,Phi_olr_nh,linewidth=3,color=green_color,\
    label='Latitude of OLR 250W/m^2 cutoff latitude from monthly OLR')
plt.plot(time,Phi_olrcs_nh,linewidth=1,color=tuple([0.5*x for x in green_color]),\
    label='Latitude of OLR 250W/m^2 cutoff latitude from monthly clear-sky OLR')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr_nh_ANN,linewidth=3,color=blue_color,\
    label='Latitude of OLR 250W/m^2 cutoff latitude from annual mean OLR')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olrcs_nh_ANN,linewidth=1,color=tuple([0.5*x for x in blue_color]),\
    label='Latitude of OLR 250W/m^2 cutoff latitude from annual mean clear-sky OLR')
plt.ylabel('NH OLR cutoff latitude')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(time,Phi_olr_sh,linewidth=3,color=green_color)
plt.plot(time,Phi_olrcs_sh,linewidth=1,color=tuple([0.5*x for x in green_color]))
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr_sh_ANN,linewidth=3,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olrcs_sh_ANN,linewidth=1,color=tuple([0.5*x for x in blue_color]))
plt.xlabel('Year')
plt.ylabel('SH OLR cutoff latitude')
plt.show()

plt.figure(5)
plt.subplot(211)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr_nh_ANN,linewidth=3,color=tuple([0.5*x for x in blue_color]),\
    label='Latitude of OLR 250W/m^2 {default} cutoff latitude from annual-mean OLR')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr240_nh_ANN,linewidth=3,color=blue_color,\
    label='Latitude of OLR 240W/m^2 cutoff latitude from annual-mean OLR')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr20_nh_ANN,linewidth=3,color=green_color,\
    label='Latitude of OLR -20W/m^2 cutoff latitude from annual-mean OLR')
plt.ylabel('NH OLR cutoff latitude')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr_sh_ANN,linewidth=3,color=tuple([0.5*x for x in blue_color]))
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr240_sh_ANN,linewidth=3,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_olr20_sh_ANN,linewidth=3,color=green_color)
plt.xlabel('Year')
plt.ylabel('SH OLR cutoff latitude')
plt.show()


## 4) STJ -- Subtropical Jet (STJ) latitude
#read zonal wind U(time,lat,lev), latitude and level
f_U = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ua.nc'),'r')
U = f_U.variables['ua'][:]
lat = f_U.variables['lat'][:]
lev = f_U.variables['lev'][:]
f_U.close()

#Change axes of u to be [time, lat]
U = np.transpose(U, (2,1,0))

# Calculate STJ latitude from annual mean
U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=np.arange(12))

Phi_stj_nh_ANN_adj = np.zeros((np.shape(U_ANN)[0],))  # latitude of NH STJ from annual mean U
Phi_stj_sh_ANN_adj = np.zeros((np.shape(U_ANN)[0],))  # latitude of SH STJ from annual mean U
Phi_stj_nh_ANN_core = np.zeros((np.shape(U_ANN)[0],)) # latitude of NH STJ from annual mean U
Phi_stj_sh_ANN_core = np.zeros((np.shape(U_ANN)[0],)) # latitude of SH STJ from annual mean U

for j in range(np.shape(U_ANN)[0]):
  # Default method =  'adjusted'
  Phi_stj_sh_ANN_adj [j], Phi_stj_nh_ANN_adj[j] = pyt.TropD_Metric_STJ(U_ANN[j,:,:], lat, lev)
  Phi_stj_sh_ANN_core[j], Phi_stj_nh_ANN_core[j] = pyt.TropD_Metric_STJ(U_ANN[j,:,:], lat, lev, method='core')


plt.figure(6)
plt.subplot(211)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_stj_nh_ANN_adj,linewidth=2,color=green_color,\
    label='Latitude of STJ from anual mean U, using \'adjusted\' method')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_stj_nh_ANN_core,linewidth=2,color=blue_color,\
    label='Latitude of STJ from anual mean U, using \'core\' method')
plt.ylabel('NH STJ latitude')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_stj_sh_ANN_adj,linewidth=2,color=green_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_stj_sh_ANN_core,linewidth=2,color=blue_color)
plt.xlabel('Year')
plt.ylabel('SH STJ latitude')
plt.show()

## 5) EDJ -- Eddy Driven Jet (EDJ) latitude
#read zonal wind U(time,lat,lev), latitude and level
f_U = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ua.nc'),'r')
U = f_U.variables['ua'][:]
lat = f_U.variables['lat'][:]
lev = f_U.variables['lev'][:]
f_U.close()


#Change axes of u to be [time, lat]
U = np.transpose(U, (2,1,0))

Phi_edj_nh = np.zeros((np.shape(U)[0],)) # latitude of monthly NH EDJ
Phi_edj_sh = np.zeros((np.shape(U)[0],)) # latitude of monthly SH EDJ

for j in range(np.shape(U)[0]):
  Phi_edj_sh[j], Phi_edj_nh[j] = pyt.TropD_Metric_EDJ(U[j,:,:,] ,lat, lev, method='max')

# Calculate EDJ latitude from annual mean
U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=np.arange(12))

Phi_edj_nh_ANN = np.zeros((np.shape(U_ANN)[0],)) # latitude of NH EDJ from annual mean U  
Phi_edj_sh_ANN = np.zeros((np.shape(U_ANN)[0],)) # latitude of SH EDJ from annual mean U

for j in range(np.shape(U_ANN)[0]):
  Phi_edj_sh_ANN[j], Phi_edj_nh_ANN[j] = pyt.TropD_Metric_EDJ(U_ANN[j,:,:], lat, lev)

plt.figure(7)
plt.subplot(211)
plt.plot(time,Phi_edj_nh,linewidth=1,color=green_color,\
    label='Latitude of EDJ from monthly mean U')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_edj_nh_ANN,linewidth=2,color=blue_color,\
    label='Latitude of EDJ from annual mean U')
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_edj_nh, season=np.arange(12)),color='k',linewidth=2,\
    label='Latitude of EDJ from annual mean of monthly metric values')
plt.ylabel('NH EDJ latitude')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(time,Phi_edj_sh,linewidth=1,color=green_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_edj_sh_ANN,linewidth=2,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_edj_sh, season=np.arange(12)),color='k',linewidth=2)
plt.xlabel('Year')
plt.ylabel('SH EDJ latitude')
plt.show()


## 6) PE -- Precipitation minus evaporation subtropical zero crossing latitude
# read zonal mean monthly precipitation pr(time,lat)
f_pr = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/pr.nc'),'r')
# read zonal mean monthly evaporation drived from surface latent heat flux hfls(time,lat)
f_er = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/hfls.nc'),'r')
#Latent heat of vaporization
L = 2510400.0

pr = f_pr.variables['pr'][:]
er = -f_er.variables['hfls'][:]/L
PE = pr - er 

lat = f_pr.variables['lat'][:]
f_pr.close()
f_er.close()


#Change axes of pr and er to be [time, lat]
PE = np.transpose(PE, (1,0))

PE_ANN = pyt.TropD_Calculate_Mon2Season(PE, season=np.arange(12))

Phi_pe_nh = np.zeros((np.shape(PE)[0],))         # latitude of NH PminusE metric
Phi_pe_sh = np.zeros((np.shape(PE)[0],))         # latitude of SH PminusE metri

Phi_pe_nh_ANN = np.zeros((np.shape(PE_ANN)[0],)) # latitude of NH PminusE metric from annual mean PminusE
Phi_pe_sh_ANN = np.zeros((np.shape(PE_ANN)[0],)) # latitude of SH PminusE metric from annual mean PminusE

for j in range(np.shape(PE)[0]):
  # Default method = 'zero_crossing'
  Phi_pe_sh[j], Phi_pe_nh[j] = pyt.TropD_Metric_PE(PE[j,:], lat)

for j in range(np.shape(PE_ANN)[0]):
  Phi_pe_sh_ANN[j], Phi_pe_nh_ANN[j] = pyt.TropD_Metric_PE(PE_ANN[j,:], lat)

plt.figure(8)
plt.subplot(211)
plt.plot(time,Phi_pe_nh,linewidth=2,color=green_color,\
    label='Latitude of P minus E zero-crossing')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_pe_nh_ANN,linewidth=2,color=blue_color,\
    label='Latitude of P minus E zero-crossing from annual mean field')
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_pe_nh, season=np.arange(12)),color='k',linewidth=2,\
    label='Latitude of P minus E zero-crossing from annual mean of monthly metric')
plt.ylabel('NH P - E zero-crossing')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(time,Phi_pe_sh,linewidth=2,color=green_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_pe_sh_ANN,linewidth=2,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_pe_sh, season=np.arange(12)),color='k',linewidth=2)
plt.xlabel('Year')
plt.ylabel('SH P - E zero-crossing')
plt.show()


## 7) UAS -- Zonal surface wind subtropical zero crossing latitude
#read zonal wind U(time,lat,lev), latitude and level
f_U = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ua.nc'),'r')
#read zonal mean surface wind U(time,lat)
f_uas = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/uas.nc'),'r')
U = f_U.variables['ua'][:]
uas = f_uas.variables['uas'][:]
lat = f_U.variables['lat'][:]
lev = f_U.variables['lev'][:]
f_U.close()


#Change axes of u to be [time, lat]
U = np.transpose(U, (2,1,0))
uas = np.transpose(uas, (1,0))

uas_ANN = pyt.TropD_Calculate_Mon2Season(uas, season=np.arange(12))
U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=np.arange(12))

Phi_uas_nh = np.zeros((np.shape(uas)[0],))         # latitude of NH surface zonal wind metric                        
Phi_uas_sh = np.zeros((np.shape(uas)[0],))         # latitude of SH surface zonal wind metric

Phi_uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],)) # latitude of NH surface zonal wind metric from annual mean surface zonal wind
Phi_uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],)) # latitude of SH surface zonal wind metric from annual mean surface zonal wind
Phi_Uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],)) # latitude of NH surface zonal wind metric from annual mean zonal wind (lat,lev)
Phi_Uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],)) # latitude of SH surface zonal wind metric from annual mean zonal wind (lat,lev)

for j in range(np.shape(uas)[0]):
  Phi_uas_sh[j], Phi_uas_nh[j] = pyt.TropD_Metric_UAS(uas[j,:], lat)

for j in range(np.shape(uas_ANN)[0]):
  Phi_uas_sh_ANN[j], Phi_uas_nh_ANN[j] = pyt.TropD_Metric_UAS(uas_ANN[j,:], lat)
  Phi_Uas_sh_ANN[j], Phi_Uas_nh_ANN[j] = pyt.TropD_Metric_UAS(U_ANN[j,:], lat, lev)


plt.figure(9)
plt.subplot(211)
plt.plot(time,Phi_uas_nh,linewidth=2,color=green_color,\
    label='Latitude of surface zonal wind zero crossing')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_uas_nh_ANN,linewidth=2,color=blue_color,\
    label='Latitude of surface zonal wind zero crossing from annual mean field')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_Uas_nh_ANN,linewidth=2,color=red_color,\
    label='Latitude of 850 hPa zonal wind zero crossing from annual mean field')
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_uas_nh, season=np.arange(12)),color='k',linewidth=2,\
    label='Latitude of surface zonal wind zero crossing from annual mean of monthly metric')
plt.ylabel('NH uas zero-crossing')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(time,Phi_uas_sh,linewidth=2,color=green_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_uas_sh_ANN,linewidth=2,color=blue_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_Uas_sh_ANN,linewidth=2,color=red_color)
plt.plot(np.arange(y1,y2+1) + 0.5,pyt.TropD_Calculate_Mon2Season(Phi_uas_sh, season=np.arange(12)),color='k',linewidth=2)
plt.xlabel('Year')
plt.ylabel('SH uas zero-crossing')
plt.show()

## 8) PSL -- Sea-level Pressure Maximum
# read sea-level pressure ps(time,lat) and latitude
f_ps = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/psl.nc'),'r')
ps = f_ps.variables['psl'][:]
lat = f_ps.variables['lat'][:]
f_ps.close()


#Change axes of ps to be [time, lat]
ps = np.transpose(ps, (1,0))

ps_DJF = pyt.TropD_Calculate_Mon2Season(ps, season=np.array([0,1,11])) # calculate DJF means
ps_JJA = pyt.TropD_Calculate_Mon2Season(ps, season=np.array([5,6,7]))  # calculate JJA means

Phi_ps_DJF_nh = np.zeros((np.shape(ps_DJF)[0],)) # latitude of monthly NH max surface pressure
Phi_ps_JJA_nh = np.zeros((np.shape(ps_JJA)[0],)) # latitude of monthly NH max surface pressure
Phi_ps_DJF_sh = np.zeros((np.shape(ps_DJF)[0],)) # latitude of monthly SH max surface pressure
Phi_ps_JJA_sh = np.zeros((np.shape(ps_JJA)[0],)) # latitude of monthly SH max surface pressure

for j in range(np.shape(ps_DJF)[0]):
  # Default method = 'max'
  Phi_ps_DJF_sh[j], Phi_ps_DJF_nh[j] = pyt.TropD_Metric_PSL(ps_DJF[j,:], lat)

for j in range(np.shape(ps_JJA)[0]):
  Phi_ps_JJA_sh[j], Phi_ps_JJA_nh[j] = pyt.TropD_Metric_PSL(ps_JJA[j,:], lat)

plt.figure(10)
plt.subplot(211)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_ps_DJF_nh,linewidth=2,color=green_color,\
    label='Latitude of max sea-level pressure during DJF')
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_ps_JJA_nh,linewidth=2,color=blue_color,\
    label='Latitude of max sea-level pressure during JJA')
plt.ylabel('NH max psl latitude')
plt.legend(loc='best', frameon=False)
plt.subplot(212)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_ps_DJF_sh,linewidth=2,color=green_color)
plt.plot(np.arange(y1,y2+1) + 0.5,Phi_ps_JJA_sh,linewidth=2,color=blue_color)
plt.ylabel('SH max psl latitude')
plt.show()

## 9) Compare annual mean metrics
#Psi500
#read meridional velocity V(time,lat,lev), latitude and level
f_V = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/va.nc'),'r')
V = f_V.variables['va'][:]
lat = f_V.variables['lat'][:]
lev = f_V.variables['lev'][:]
f_V.close()

#Change axes of V to be [time, lat]
V = np.transpose(V, (2,1,0))

V_ANN = pyt.TropD_Calculate_Mon2Season(V, season=np.arange(12))

Phi_psi_nh_ANN = np.zeros((np.shape(V_ANN)[0],))
Phi_psi_sh_ANN = np.zeros((np.shape(V_ANN)[0],))

for j in range(np.shape(V_ANN)[0]):
  Phi_psi_sh_ANN[j], Phi_psi_nh_ANN[j] = pyt.TropD_Metric_PSI(V_ANN[j,:,:], lat, lev)

# Tropopause break
#read meridional temperature T(time,lat,lev), latitude and level
f_T = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ta.nc'),'r')
T = f_T.variables['ta'][:]
f_T.close()

#Change axes of T to be [time, lat]
T = np.transpose(T, (2,1,0))

T_ANN = pyt.TropD_Calculate_Mon2Season(T, season=np.arange(12))

Phi_tpb_nh_ANN = np.zeros((np.shape(T_ANN)[0],))
Phi_tpb_sh_ANN = np.zeros((np.shape(T_ANN)[0],))

for j in range(np.shape(T_ANN)[0]):
  Phi_tpb_sh_ANN[j], Phi_tpb_nh_ANN[j] = pyt.TropD_Metric_TPB(T_ANN[j,:,:], lat, lev)

# Surface pressure max
# read sea-level pressure ps(time,lat) and latitude
f_ps = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/psl.nc'),'r')
ps = f_ps.variables['psl'][:]
f_ps.close()

#Change axes of ps to be [time, lat]
ps = np.transpose(ps, (1,0))

ps_ANN = pyt.TropD_Calculate_Mon2Season(ps, season=np.arange(12))

Phi_ps_nh_ANN = np.zeros((np.shape(ps_ANN)[0],))
Phi_ps_sh_ANN = np.zeros((np.shape(ps_ANN)[0],))

for j in range(np.shape(ps_ANN)[0]):
  Phi_ps_sh_ANN[j], Phi_ps_nh_ANN[j] = pyt.TropD_Metric_PSL(ps_ANN[j,:], lat)

# Eddy driven jet
#read zonal wind U(time,lat,lev), latitude and level
f_U = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ua.nc'),'r')
U = f_U.variables['ua'][:]
f_U.close()

#Change axes of U to be [time, lat]
U = np.transpose(U, (2,1,0))

U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=np.arange(12))

Phi_edj_nh_ANN = np.zeros((np.shape(U_ANN)[0],))
Phi_edj_sh_ANN = np.zeros((np.shape(U_ANN)[0],))

for j in range(np.shape(U_ANN)[0]):
  Phi_edj_sh_ANN[j], Phi_edj_nh_ANN[j] = pyt.TropD_Metric_EDJ(U_ANN[j,:,:], lat, lev)

# Subtropical jet
Phi_stj_nh_ANN = np.zeros((np.shape(U_ANN)[0],))
Phi_stj_sh_ANN = np.zeros((np.shape(U_ANN)[0],))

for j in range(np.shape(U_ANN)[0]):
  Phi_stj_sh_ANN[j], Phi_stj_nh_ANN[j] = pyt.TropD_Metric_STJ(U_ANN[j,:,:], lat, lev)

# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
f_olr = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/rlnt.nc'),'r')
olr = -f_olr.variables['rlnt'][:]
f_olr.close()

#Change axes of olr to be [time, lat]
olr = np.transpose(olr, (1,0))

olr_ANN = pyt.TropD_Calculate_Mon2Season(olr, season=np.arange(12))

Phi_olr_nh_ANN = np.zeros((np.shape(olr_ANN)[0],))
Phi_olr_sh_ANN = np.zeros((np.shape(olr_ANN)[0],))

for j in range(np.shape(olr_ANN)[0]):
  Phi_olr_sh_ANN[j], Phi_olr_nh_ANN[j] = pyt.TropD_Metric_OLR(olr_ANN[j,:], lat)

# P minus E
# read zonal mean monthly precipitation pr(time,lat)
f_pr = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/pr.nc'),'r')
pr = f_pr.variables['pr'][:]
f_pr.close()

#Latent heat of vaporization
L = 2510400.0

# read zonal mean monthly evaporation drived from surface latent heat flux hfls(time,lat)
f_er = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/hfls.nc'),'r')
er = -f_er.variables['hfls'][:]/L
f_er.close()

#Change axes of ps and er to be [time, lat]
pr = np.transpose(pr, (1,0))
er = np.transpose(er, (1,0))

PE = pr - er
PE_ANN = pyt.TropD_Calculate_Mon2Season(PE, season=np.arange(12))

Phi_pe_nh_ANN = np.zeros((np.shape(PE_ANN)[0],))

Phi_pe_sh_ANN = np.zeros((np.shape(PE_ANN)[0],))

for j in range(np.shape(PE_ANN)[0]):
  Phi_pe_sh_ANN[j], Phi_pe_nh_ANN[j] = pyt.TropD_Metric_PE(PE_ANN[j,:], lat)

# Surface winds
#read zonal mean surface wind U(time,lat)
f_uas = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/uas.nc'),'r')
uas = f_uas.variables['uas'][:]
f_uas.close()

#Change axes of uas to be [time, lat]
uas = np.transpose(uas, (1,0))

uas_ANN = pyt.TropD_Calculate_Mon2Season(uas, season=np.arange(12))

Phi_uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],))

Phi_uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],))

for j in range(np.shape(uas_ANN)[0]):
  Phi_uas_sh_ANN[j], Phi_uas_nh_ANN[j] = pyt.TropD_Metric_UAS(uas_ANN[j,:], lat)


plt.figure(11)
plt.subplot(211)
plt.plot(np.arange(y1,y2+1),Phi_psi_nh_ANN,linewidth=2,color=tuple([0,0,0]),\
    label='PSI')
plt.plot(np.arange(y1,y2+1),Phi_tpb_nh_ANN,linewidth=2,color=green_color,\
    label='TPB')
plt.plot(np.arange(y1,y2+1),Phi_edj_nh_ANN,linewidth=2,color=blue_color,\
    label='EDJ')
plt.plot(np.arange(y1,y2+1),Phi_stj_nh_ANN,linewidth=2,color=red_color,\
    label='STJ')
plt.plot(np.arange(y1,y2+1),Phi_olr_nh_ANN,linewidth=2,color=lightblue_color,\
    label='OLR')
plt.plot(np.arange(y1,y2+1),Phi_pe_nh_ANN,linewidth=2,color=orange_color,\
    label='P-E')
plt.plot(np.arange(y1,y2+1),Phi_uas_nh_ANN,linewidth=2,color=purple_color,\
    label='UAS')
plt.plot(np.arange(y1,y2+1),Phi_ps_nh_ANN,linestyle='--',linewidth=2,color=maroon_color,\
    label='PSL')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
plt.ylabel('NH HC edge')
plt.subplot(212)
plt.plot(np.arange(y1,y2+1),Phi_psi_sh_ANN,linewidth=2,color=tuple([0,0,0]),\
    label='PSI')
plt.plot(np.arange(y1,y2+1),Phi_tpb_sh_ANN,linewidth=2,color=green_color,\
    label='TPB')
plt.plot(np.arange(y1,y2+1),Phi_edj_sh_ANN,linewidth=2,color=blue_color,\
    label='EDJ')
plt.plot(np.arange(y1,y2+1),Phi_stj_sh_ANN,linewidth=2,color=red_color,\
    label='STJ')
plt.plot(np.arange(y1,y2+1),Phi_olr_sh_ANN,linewidth=2,color=lightblue_color,\
    label='OLR')
plt.plot(np.arange(y1,y2+1),Phi_pe_sh_ANN,linewidth=2,color=orange_color,\
    label='P-E')
plt.plot(np.arange(y1,y2+1),Phi_uas_sh_ANN,linewidth=2,color=purple_color,\
    label='UAS')
plt.plot(np.arange(y1,y2+1),Phi_ps_sh_ANN,linestyle='--',linewidth=2,color=maroon_color,\
    label='PSL')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
plt.xlabel('Year')
plt.ylabel('SH HC edge')
plt.show()


## 10) Validate metrics
# Check calculations with precalculated values from file within roundoff error
#Psi500
#read meridional velocity V(time,lat,lev), latitude and level
f_V = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/va.nc'),'r')
V = f_V.variables['va'][:]
lat = f_V.variables['lat'][:]
lev = f_V.variables['lev'][:]
f_V.close()


#Change axes of V to be [time, lat]
V = np.transpose(V, (2,1,0))

V_ANN = pyt.TropD_Calculate_Mon2Season(V, season=np.arange(12))

Phi_psi_nh = np.zeros((np.shape(V)[0],))
Phi_psi_sh = np.zeros((np.shape(V)[0],))

Phi_psi_nh_ANN = np.zeros((np.shape(V_ANN)[0],))
Phi_psi_sh_ANN = np.zeros((np.shape(V_ANN)[0],))

for j in range(np.shape(V)[0]):
  Phi_psi_sh[j], Phi_psi_nh[j] = pyt.TropD_Metric_PSI(V[j,:,:], lat, lev)

for j in range(np.shape(V_ANN)[0]):
  Phi_psi_sh_ANN[j], Phi_psi_nh_ANN[j] = pyt.TropD_Metric_PSI(V_ANN[j,:,:], lat, lev)


f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PSI_ANN.nc'),'r')
Phi_nh = f_Phi.variables['PSI_NH'][:]
Phi_sh = f_Phi.variables['PSI_SH'][:]
f_Phi.close()


if not (np.std(Phi_nh - Phi_psi_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_psi_sh_ANN) < 1e-10):
  print('Warning: annual-mean Validation and calculated PSI metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated PSI metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PSI.nc'),'r')
Phi_nh = f_Phi.variables['PSI_NH'][:]
Phi_sh = f_Phi.variables['PSI_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_psi_nh) < 1e-10 and np.std(Phi_sh - Phi_psi_sh) < 1e-10):
  print('Warning: monthly Validation and calculated PSI metrics are NOT equal!')
else:
  print('OK. Monthly Validation and calculated PSI metrics are the same!')

# Tropopause break
#read temperature T(time,lat,lev), latitude and level
f_T = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ta.nc'),'r')
T = f_T.variables['ta'][:]
f_T.close()

#Change axes of T to be [time, lat, lev]
T = np.transpose(T, (2,1,0))

T_ANN = pyt.TropD_Calculate_Mon2Season(T, season=np.arange(12))

Phi_tpb_nh = np.zeros((np.shape(T)[0],))
Phi_tpb_sh = np.zeros((np.shape(T)[0],))

Phi_tpb_nh_ANN = np.zeros((np.shape(T_ANN)[0],))
Phi_tpb_sh_ANN = np.zeros((np.shape(T_ANN)[0],))

for j in range(np.shape(T)[0]):
  Phi_tpb_sh[j], Phi_tpb_nh[j] = pyt.TropD_Metric_TPB(T[j,:,:], lat, lev)

for j in range(np.shape(T_ANN)[0]):
  Phi_tpb_sh_ANN[j], Phi_tpb_nh_ANN[j] = pyt.TropD_Metric_TPB(T_ANN[j,:,:], lat, lev)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/TPB_ANN.nc'),'r')
Phi_nh = f_Phi.variables['TPB_NH'][:]
Phi_sh = f_Phi.variables['TPB_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_tpb_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_tpb_sh_ANN) < 1e-10):
  print('Warning: annual-mean Validation and calculated TPB metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated TPB metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/TPB.nc'),'r')
Phi_nh = f_Phi.variables['TPB_NH'][:]
Phi_sh = f_Phi.variables['TPB_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_tpb_nh) < 1e-10 and np.std(Phi_sh - Phi_tpb_sh) < 1e-10):
  print('Warning: monthly Validation and calculated TPB metrics are NOT equal!')
else:
  print('OK. Monthly Validation and calculated TPB metrics are the same!')

# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
f_ps = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/psl.nc'),'r')
ps = f_ps.variables['psl'][:]
f_ps.close()

#Change axes of ps to be [time, lat]
ps = np.transpose(ps, (1,0))

ps_DJF = pyt.TropD_Calculate_Mon2Season(ps, season=np.array([0,1,11]))
ps_MAM = pyt.TropD_Calculate_Mon2Season(ps, season=np.array([2,3,4]))
ps_JJA = pyt.TropD_Calculate_Mon2Season(ps, season=np.array([5,6,7]))
ps_SON = pyt.TropD_Calculate_Mon2Season(ps, season=np.array([8,9,10]))

Phi_ps_sh_DJF = np.zeros((np.shape(ps_DJF)[0],))
Phi_ps_sh_JJA = np.zeros((np.shape(ps_JJA)[0],))
Phi_ps_sh_MAM = np.zeros((np.shape(ps_MAM)[0],))
Phi_ps_sh_SON = np.zeros((np.shape(ps_SON)[0],))
Phi_ps_nh_DJF = np.zeros((np.shape(ps_DJF)[0],))
Phi_ps_nh_JJA = np.zeros((np.shape(ps_JJA)[0],))
Phi_ps_nh_MAM = np.zeros((np.shape(ps_MAM)[0],))
Phi_ps_nh_SON = np.zeros((np.shape(ps_SON)[0],))

for j in range(np.shape(ps_DJF)[0]):
  Phi_ps_sh_DJF[j], Phi_ps_nh_DJF[j] = pyt.TropD_Metric_PSL(ps_DJF[j,:], lat)

for j in range(np.shape(ps_JJA)[0]):
  Phi_ps_sh_JJA[j], Phi_ps_nh_JJA[j] = pyt.TropD_Metric_PSL(ps_JJA[j,:], lat)

for j in range(np.shape(ps_MAM)[0]):
  Phi_ps_sh_MAM[j], Phi_ps_nh_MAM[j] = pyt.TropD_Metric_PSL(ps_MAM[j,:], lat)

for j in range(np.shape(ps_SON)[0]):
  Phi_ps_sh_SON[j], Phi_ps_nh_SON[j] = pyt.TropD_Metric_PSL(ps_SON[j,:], lat)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PSL_DJF.nc'),'r')
Phi_nh = f_Phi.variables['PSL_NH'][:]
Phi_sh = f_Phi.variables['PSL_SH'][:]
f_Phi.close()

if not (np.std(Phi_sh - Phi_ps_sh_DJF) < 1e-10) or not (np.std(Phi_nh - Phi_ps_nh_DJF) < 1e-10):
  print('Warning: DJF Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. DJF Validation and calculated PSL metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PSL_JJA.nc'),'r')
Phi_nh = f_Phi.variables['PSL_NH'][:]
Phi_sh = f_Phi.variables['PSL_SH'][:]
f_Phi.close()

if not (np.std(Phi_sh - Phi_ps_sh_JJA) < 1e-10) or not (np.std(Phi_nh - Phi_ps_nh_JJA) < 1e-10):
  print('Warning: JJA Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. JJA Validation and calculated PSL metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PSL_MAM.nc'),'r')
Phi_nh = f_Phi.variables['PSL_NH'][:]
Phi_sh = f_Phi.variables['PSL_SH'][:]
f_Phi.close()

if not (np.std(Phi_sh - Phi_ps_sh_MAM) < 1e-10) or not (np.std(Phi_nh - Phi_ps_nh_MAM) < 1e-10):
  print('Warning: MAM Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. MAM Validation and calculated PSL metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PSL_SON.nc'),'r')
Phi_nh = f_Phi.variables['PSL_NH'][:]
Phi_sh = f_Phi.variables['PSL_SH'][:]
f_Phi.close()

if not (np.std(Phi_sh - Phi_ps_sh_SON) < 1e-10) or not (np.std(Phi_nh - Phi_ps_nh_SON) < 1e-10):
  print('Warning: SON Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. SON Validation and calculated PSL metrics are the same!')

# Eddy driven jet
#read zonal wind U(time,lat,lev), latitude and level
f_U = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/ua.nc'),'r')
U = f_U.variables['ua'][:]
f_U.close()

#Change axes of U to be [time, lat]
U = np.transpose(U, (2,1,0))

U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=np.arange(12))

Phi_edj_nh=np.zeros((np.shape(U)[0],))
Phi_edj_sh=np.zeros((np.shape(U)[0],))

Phi_edj_nh_ANN = np.zeros((np.shape(U_ANN)[0],))
Phi_edj_sh_ANN = np.zeros((np.shape(U_ANN)[0],))

for j in range(np.shape(U)[0]):
  Phi_edj_sh[j], Phi_edj_nh[j] = pyt.TropD_Metric_EDJ(U[j,:,:], lat, lev)

for j in range(np.shape(U_ANN)[0]):
  Phi_edj_sh_ANN[j], Phi_edj_nh_ANN[j] = pyt.TropD_Metric_EDJ(U_ANN[j,:,:], lat, lev)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/EDJ_ANN.nc'),'r')
Phi_nh = f_Phi.variables['EDJ_NH'][:]
Phi_sh = f_Phi.variables['EDJ_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_edj_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_edj_sh_ANN) < 1e-10):
  print('Warning: annual-mean Validation and calculated EDJ metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated EDJ metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/EDJ.nc'),'r')
Phi_nh = f_Phi.variables['EDJ_NH'][:]
Phi_sh = f_Phi.variables['EDJ_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_edj_nh) < 1e-10 and np.std(Phi_sh - Phi_edj_sh) < 1e-10):
  print('Warning: monthly Validation and calculated EDJ metrics are NOT equal!')
else:
  print('OK. Monthly Validation and calculated EDJ metrics are the same!')

# Subtropical jet
Phi_stj_nh = np.zeros((np.shape(U)[0],))
Phi_stj_sh = np.zeros((np.shape(U)[0],))
Phi_stj_nh_ANN = np.zeros((np.shape(U_ANN)[0],))
Phi_stj_sh_ANN = np.zeros((np.shape(U_ANN)[0],))

for j in range(np.shape(U)[0]):
  Phi_stj_sh[j], Phi_stj_nh[j] = pyt.TropD_Metric_STJ(U[j,:,:], lat, lev)

for j in range(np.shape(U_ANN)[0]):
  Phi_stj_sh_ANN[j], Phi_stj_nh_ANN[j] = pyt.TropD_Metric_STJ(U_ANN[j,:,:], lat, lev)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/STJ_ANN.nc'),'r')
Phi_nh = f_Phi.variables['STJ_NH'][:]
Phi_sh = f_Phi.variables['STJ_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_stj_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_stj_sh_ANN) < 1e-10):
  print('Warning: annual-mean Validation and calculated STJ metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated STJ metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/STJ.nc'),'r')
Phi_nh = f_Phi.variables['STJ_NH'][:]
Phi_sh = f_Phi.variables['STJ_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_stj_nh) < 1e-10 and np.std(Phi_sh - Phi_stj_sh) < 1e-10):
  print('Warning: monthly Validation and calculated STJ metrics are NOT equal!')
else:
  print('OK. Monthly Validation and calculated STJ metrics are the same!')

# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
f_olr = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/rlnt.nc'),'r')
olr = -f_olr.variables['rlnt'][:]
f_olr.close()

#Change axes of olr to be [time, lat]
olr = np.transpose(olr, (1,0))

olr_ANN = pyt.TropD_Calculate_Mon2Season(olr, season=np.arange(12))

Phi_olr_nh = np.zeros((np.shape(olr)[0],))
Phi_olr_sh = np.zeros((np.shape(olr)[0],))
Phi_olr_nh_ANN = np.zeros((np.shape(olr_ANN)[0],))
Phi_olr_sh_ANN = np.zeros((np.shape(olr_ANN)[0],))

for j in range(np.shape(olr)[0]):
  Phi_olr_sh[j], Phi_olr_nh[j] = pyt.TropD_Metric_OLR(olr[j,:], lat)

for j in range(np.shape(olr_ANN)[0]):
  Phi_olr_sh_ANN[j], Phi_olr_nh_ANN[j] = pyt.TropD_Metric_OLR(olr_ANN[j,:], lat)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/OLR_ANN.nc'),'r')
Phi_nh = f_Phi.variables['OLR_NH'][:]
Phi_sh = f_Phi.variables['OLR_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_olr_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_olr_sh_ANN) < 1e-10):
  print 'Warning: annual-mean Validation and calculated OLR metrics are NOT equal!'
else:
  print 'OK. Annual-mean Validation and calculated OLR metrics are the same!'

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/OLR.nc'),'r')
Phi_nh = f_Phi.variables['OLR_NH'][:]
Phi_sh = f_Phi.variables['OLR_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_olr_nh) < 1e-10 and np.std(Phi_sh - Phi_olr_sh) < 1e-10):
  print 'Warning: monthly Validation and calculated OLR metrics are NOT equal!'
else:
  print 'OK. Monthly Validation and calculated OLR metrics are the same!'

# P minus E
# read zonal mean monthly precipitation pr(time,lat)
f_pr = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/pr.nc'),'r')
pr = f_pr.variables['pr'][:]
f_pr.close()

#Latent heat of vaporization
L = 2510400.0

# read zonal mean monthly evaporation drived from surface latent heat flux hfls(time,lat)
f_er = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/hfls.nc'),'r')
er = -f_er.variables['hfls'][:]/L
f_er.close()

#Change axes of ps and er to be [time, lat]
pr = np.transpose(pr, (1,0))
er = np.transpose(er, (1,0))

PE = pr - er
PE_ANN = pyt.TropD_Calculate_Mon2Season(PE, season=np.arange(12))

Phi_pe_nh=np.zeros((np.shape(PE)[0],))
Phi_pe_sh=np.zeros((np.shape(PE)[0],))

Phi_pe_nh_ANN = np.zeros((np.shape(PE_ANN)[0],))
Phi_pe_sh_ANN = np.zeros((np.shape(PE_ANN)[0],))

for j in range(np.shape(PE)[0]):
  Phi_pe_sh[j], Phi_pe_nh[j] = pyt.TropD_Metric_PE(PE[j,:], lat)

for j in range(np.shape(PE_ANN)[0]):
  Phi_pe_sh_ANN[j], Phi_pe_nh_ANN[j] = pyt.TropD_Metric_PE(PE_ANN[j,:], lat)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PE_ANN.nc'),'r')
Phi_nh = f_Phi.variables['PE_NH'][:]
Phi_sh = f_Phi.variables['PE_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_pe_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_pe_sh_ANN) < 1e-10):
  print('Warning: annual-mean Validation and calculated P-E metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated P-E metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/PE.nc'),'r')
Phi_nh = f_Phi.variables['PE_NH'][:]
Phi_sh = f_Phi.variables['PE_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_pe_nh) < 1e-10 and np.std(Phi_sh - Phi_pe_sh) < 1e-10):
  print('Warning: monthly Validation and calculated P-E metrics are NOT equal!')
else:
  print('OK. Monthly Validation and calculated P-E metrics are the same!')

# Surface winds
#read zonal mean surface wind U(time,lat)
f_uas = netcdf.netcdf_file(os.path.join(dirname, '../ValidationData/uas.nc'),'r')
uas = f_uas.variables['uas'][:]
f_uas.close()

#Change axes of uas to be [time, lat]
uas = np.transpose(uas, (1,0))

uas_ANN = pyt.TropD_Calculate_Mon2Season(uas, season=np.arange(12))

Phi_uas_nh=np.zeros((np.shape(uas)[0],))
Phi_uas_sh=np.zeros((np.shape(uas)[0],))

Phi_uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],))
Phi_uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],))

for j in range(np.shape(uas)[0]):
  Phi_uas_sh[j], Phi_uas_nh[j] = pyt.TropD_Metric_UAS(uas[j,:], lat)

for j in range(np.shape(uas_ANN)[0]):
  Phi_uas_sh_ANN[j], Phi_uas_nh_ANN[j] = pyt.TropD_Metric_UAS(uas_ANN[j,:], lat)

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/UAS_ANN.nc'),'r')
Phi_nh = f_Phi.variables['UAS_NH'][:]
Phi_sh = f_Phi.variables['UAS_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_uas_nh_ANN) < 1e-10 and np.std(Phi_sh - Phi_uas_sh_ANN) < 1e-10):
  print('Warning: annual-mean Validation and calculated UAS metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated UAS metrics are the same!')

f_Phi = netcdf.netcdf_file(os.path.join(dirname, '../ValidationMetrics/UAS.nc'),'r')
Phi_nh = f_Phi.variables['UAS_NH'][:]
Phi_sh = f_Phi.variables['UAS_SH'][:]
f_Phi.close()

if not (np.std(Phi_nh - Phi_uas_nh) < 1e-10 and np.std(Phi_sh - Phi_uas_sh) < 1e-10):
  print('Warning: monthly Validation and calculated UAS metrics are NOT equal!')
else:
  print('OK. Monthly Validation and calculated UAS metrics are the same!')

