import numpy as np
import xarray as xr
import pandas as pd
import xarray_metrics as pyt_metrics
from scipy import interpolate

##Validate metrics
# Check calculations with precalculated values from file within roundoff error
#Psi500
#read meridional velocity V(time,lat,lev), latitude and level
 
v_data = xr.open_dataset('./ValidationData/va.nc')

#Define a time axis
v_data['time'] = pd.date_range(start='1979-01-01', periods=v_data.sizes['time'], freq='MS')    

#yearly mean of data
v_annual = v_data.groupby('time.year').mean('time')

psi_metrics = v_annual.pyt_metrics.psi()         

psi_metric_validation_data = xr.open_dataset('./ValidationMetrics/PSI_ANN.nc')         

if not (np.max(psi_metrics.psi.sel(metrics=0).values - psi_metric_validation_data.PSI_SH.values)) < 1e-10 and \
        (np.max(psi_metrics.psi.sel(metrics=1).values - psi_metric_validation_data.PSI_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated PSI metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated PSI metrics are the same!')



# Tropopause break
#read temperature T(time,lat,lev), latitude and level
T_data = xr.open_dataset('./ValidationData/ta.nc')

#Define a time axis
T_data['time'] = pd.date_range(start='1979-01-01', periods=T_data.sizes['time'], freq='MS')    

#yearly mean of data
T_annual = T_data.groupby('time.year').mean('time')

tpb_metrics = T_annual.pyt_metrics.tpb()         

tpb_metric_validation_data = xr.open_dataset('./ValidationMetrics/TPB_ANN.nc')         

if not (np.max(tpb_metrics.tpb.sel(metrics=0).values - tpb_metric_validation_data.TPB_SH.values)) < 1e-10 and \
        (np.max(tpb_metrics.tpb.sel(metrics=1).values - tpb_metric_validation_data.TPB_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated TPB metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated TPB metrics are the same!')


# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
psl_data = xr.open_dataset('./ValidationData/psl.nc')                                                                

psl_data['time'] = pd.date_range(start='1979-01-01', periods=psl_data.sizes['time'], freq='MS')                      

psl_DJF_select = psl_data.sel(time=psl_data.time.dt.season=="DJF") 
psl_MAM_select = psl_data.sel(time=psl_data.time.dt.season=="MAM") 
psl_JJA_select = psl_data.sel(time=psl_data.time.dt.season=="JJA") 
psl_SON_select = psl_data.sel(time=psl_data.time.dt.season=="SON") 

# calculate mean per year 
psl_DJF = psl_DJF_select.groupby(psl_DJF_select.time.dt.year).mean("time")    
psl_MAM = psl_MAM_select.groupby(psl_MAM_select.time.dt.year).mean("time")    
psl_JJA = psl_JJA_select.groupby(psl_JJA_select.time.dt.year).mean("time")    
psl_SON = psl_SON_select.groupby(psl_SON_select.time.dt.year).mean("time")    

psl_DJF_metrics = psl_DJF.pyt_metrics.psl()   
psl_MAM_metrics = psl_MAM.pyt_metrics.psl()   
psl_JJA_metrics = psl_JJA.pyt_metrics.psl()   
psl_SON_metrics = psl_SON.pyt_metrics.psl()   

psl_DJF_metric_validation_data = xr.open_dataset('./ValidationMetrics/PSL_DJF.nc')    
psl_MAM_metric_validation_data = xr.open_dataset('./ValidationMetrics/PSL_MAM.nc')    
psl_JJA_metric_validation_data = xr.open_dataset('./ValidationMetrics/PSL_JJA.nc')    
psl_SON_metric_validation_data = xr.open_dataset('./ValidationMetrics/PSL_SON.nc')    

if not (np.max(psl_DJF_metrics.psl.sel(metrics=0).values - psl_DJF_metric_validation_data.PSL_SH.values)) < 1e-10 and \
        (np.max(psl_DJF_metrics.psl.sel(metrics=1).values - psl_DJF_metric_validation_data.PSL_NH.values)) < 1e-10:
  print('Warning: DJF Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. DJF Validation and calculated PSL metrics are the same!')

if not (np.max(psl_MAM_metrics.psl.sel(metrics=0).values - psl_MAM_metric_validation_data.PSL_SH.values)) < 1e-10 and \
        (np.max(psl_MAM_metrics.psl.sel(metrics=1).values - psl_MAM_metric_validation_data.PSL_NH.values)) < 1e-10:
  print('Warning: MAM Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. MAM Validation and calculated PSL metrics are the same!')

if not (np.max(psl_JJA_metrics.psl.sel(metrics=0).values - psl_JJA_metric_validation_data.PSL_SH.values)) < 1e-10 and \
        (np.max(psl_JJA_metrics.psl.sel(metrics=1).values - psl_JJA_metric_validation_data.PSL_NH.values)) < 1e-10:
  print('Warning: JJA Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. JJA Validation and calculated PSL metrics are the same!')

if not (np.max(psl_SON_metrics.psl.sel(metrics=0).values - psl_SON_metric_validation_data.PSL_SH.values)) < 1e-10 and \
        (np.max(psl_SON_metrics.psl.sel(metrics=1).values - psl_SON_metric_validation_data.PSL_NH.values)) < 1e-10:
  print('Warning: SON Validation and calculated PSL metrics are NOT equal!')
else:
  print('OK. SON Validation and calculated PSL metrics are the same!')


# Eddy driven jet
#read zonal wind U(time,lat,lev), latitude and level
u_data = xr.open_dataset('./ValidationData/ua.nc')

#Define a time axis
u_data['time'] = pd.date_range(start='1979-01-01', periods=u_data.sizes['time'], freq='MS')    

#yearly mean of data
u_annual = u_data.groupby('time.year').mean('time')

edj_metrics = u_annual.pyt_metrics.edj()         

edj_metric_validation_data = xr.open_dataset('./ValidationMetrics/EDJ_ANN.nc')         

if not (np.max(edj_metrics.edj.sel(metrics=0).values - edj_metric_validation_data.EDJ_SH.values)) < 1e-10 and \
        (np.max(edj_metrics.edj.sel(metrics=1).values - edj_metric_validation_data.EDJ_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated EDJ metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated EDJ metrics are the same!')

# Subtropical jet
#yearly mean of data
u_annual = u_data.groupby('time.year').mean('time')

stj_metrics = u_annual.pyt_metrics.stj()         

stj_metric_validation_data = xr.open_dataset('./ValidationMetrics/STJ_ANN.nc')         

if not (np.max(stj_metrics.stj.sel(metrics=0).values - stj_metric_validation_data.STJ_SH.values)) < 1e-10 and \
        (np.max(stj_metrics.stj.sel(metrics=1).values - stj_metric_validation_data.STJ_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated STJ metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated STJ metrics are the same!')


# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
olr_data = xr.open_dataset('./ValidationData/rlnt.nc')

#Define a time axis
olr_data['time'] = pd.date_range(start='1979-01-01', periods=olr_data.sizes['time'], freq='MS')    

#yearly mean of data
olr_annual = olr_data.groupby('time.year').mean('time')

olr_metrics = olr_annual.pyt_metrics.olr()         

olr_metric_validation_data = xr.open_dataset('./ValidationMetrics/OLR_ANN.nc')         

if not (np.max(olr_metrics.olr.sel(metrics=0).values - olr_metric_validation_data.OLR_SH.values)) < 1e-10 and \
        (np.max(olr_metrics.olr.sel(metrics=1).values - olr_metric_validation_data.OLR_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated OLR metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated OLR metrics are the same!')

# P minus E
# read zonal mean monthly precipitation pr(time,lat)
pr_data = xr.open_dataset('./ValidationData/pr.nc')
surf_latent_heat_flux_data = xr.open_dataset('./ValidationData/hfls.nc')

#Latent heat of vaporization
L = 2510400.0

er_data = -surf_latent_heat_flux_data/L
pe_data = (pr_data.pr - er_data.hfls).to_dataset(name='pe')

#Define a time axis
pe_data['time'] = pd.date_range(start='1979-01-01', periods=pe_data.sizes['time'], freq='MS')    

#yearly mean of data
pe_annual = pe_data.groupby('time.year').mean('time')

pe_metrics = pe_annual.pyt_metrics.pe()         

pe_metric_validation_data = xr.open_dataset('./ValidationMetrics/PE_ANN.nc')         

if not (np.max(pe_metrics.pe.sel(metrics=0).values - pe_metric_validation_data.PE_SH.values)) < 1e-10 and \
        (np.max(pe_metrics.pe.sel(metrics=1).values - pe_metric_validation_data.PE_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated PE metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated PE metrics are the same!')

# Surface winds
#read zonal mean surface wind U(time,lat)
uas_data = xr.open_dataset('./ValidationData/uas.nc')

#Define a time axis
uas_data['time'] = pd.date_range(start='1979-01-01', periods=uas_data.sizes['time'], freq='MS')    

#yearly mean of data
uas_annual = uas_data.groupby('time.year').mean('time')

uas_metrics = uas_annual.pyt_metrics.uas()         

uas_metric_validation_data = xr.open_dataset('./ValidationMetrics/UAS_ANN.nc')         

if not (np.max(uas_metrics.uas.sel(metrics=0).values - uas_metric_validation_data.UAS_SH.values)) < 1e-10 and \
        (np.max(uas_metrics.uas.sel(metrics=1).values - uas_metric_validation_data.UAS_NH.values)) < 1e-10:
  print('Warning: annual-mean Validation and calculated UAS metrics are NOT equal!')
else:
  print('OK. Annual-mean Validation and calculated UAS metrics are the same!')

