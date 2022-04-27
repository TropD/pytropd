# update to python 3, fix plots and add save option, 
# fix file-handling and reduce boilerplate - sjs 2022.01.27
import os, glob, warnings
import numpy as np
import scipy as sp
from scipy.io import netcdf
import pytropd as pyt
#don't require matplotlib but use if present
try:
    import matplotlib.pyplot as plt
    import matplotlib
    USE_MPL = True
except ImportError:
    USE_MPL = False
    
# warnings.simplefilter('default')

USE_MPL = False
#Latent heat of vaporization
L_VAP = 2510400.0

# Example codes for using the TropD package to calculate tropical width metrics
# Also validates the calculations for testing
# The code assumes that the current directory is ... pytropd/pytropd/
# (expects data in ../ValidationData and metrics in ../ValidationMetrics)

# whether to print testing summary or print detailed log
VERBOSE = False
#whether figs should be displayed or saved to ../ValidationMetrics/figs/
SAVE_FIGS = False

# some functions to reduce boilerplate
# improve file handling - sjs 2022.01.28
def get_arrs_from_nc(file,var_list):
    ''' 
    open netcdf file, construct numpy arrays from data, and close file
    arrays are returned in same order as keys in var_list
        
    ::input::
    file - path-like, full path to netcdf file to read
    var_list (list or str) - list (or single string) of variables/keys 
                                 to read from file into memory
        
    ::output::
    data_list - list of numpy arrays, same order as var_list
    '''
    
    # loop over all input, even if just one var
    if isinstance(var_list,str):
        var_list = [var_list]
    assert isinstance(var_list,list), 'expected string or list of strings'
    
    #need to open the file before passing to netcdf_file
    # to properly read data and close file
    data_list = []
    with open(file,'rb') as fbin:
        with netcdf.netcdf_file(fbin,'r') as dataset:
            for var in var_list:
                data_list.append(dataset.variables[var][:])
    
    #flatten if only one var
    if len(data_list) == 1:
        data_list = data_list[0]
    return data_list
    
# cleaner warning system - sjs 2022.01.28
def validate_against_metric(computed_arr, metric, hem, verbose=True, tol=1e-5):
    ''' 
    compare computed array for equality (within round-off error) to data
    in provided validation files and print detalied messages of results
        
    ::input::
    computed_arr - array to be validated
    metric (string) - short string corresponding to metric to validate with
    hem (string) - "NH" or "SH" corresponding to the hemisphere
    verbose (bool, optional) -  whether to display detailed messaging for
                                    checks that pass (default:True)
        
    ::output::
    (bool) - whether or not the array is valid
    '''
    
    #parse input metric str for ann vs monthly vs seasonal
    base_metric = metric.split('_')[0]
    if '_' not in metric:
        freq_str = 'Monthly'
    elif '_ANN' in metric:
        freq_str = 'Annual-mean'
    else:
        freq_str = metric.split('_')[-1]
    
    #get the validated metric
    Phi_valid = get_arrs_from_nc(metric_files[metric],f'{base_metric}_{hem}')
    
    #check if equal (within round-off)
    if np.allclose(computed_arr, Phi_valid, rtol=tol):
        if verbose: #turn off printing for passing checks (no news = good news)
            print(f'OK. {freq_str} validation and calculated {hem} '
                    f'{base_metric} metrics are the same!')
        return True    
    else:
        pct_passing = (np.isclose(computed_arr,Phi_valid,rtol=tol).sum()
                       / computed_arr.size * 100.)
        warnings.warn(f'{freq_str} validation and calculated {hem} '
                        f' {base_metric} metrics are NOT equal with '
                        f'{pct_passing:.1f}% matching',
                        category=RuntimeWarning, stacklevel=2)
        return False

# get validation files as dictionaries for easy access
# used to build absolute path to data files
dirname = os.path.dirname(__file__)

# data files
data_files = {os.path.split(f)[1].split('.')[0]: f
                for f in glob.glob(os.path.join(dirname,
                                    '../ValidationData/*.nc'))}

#metric files
metric_files = {os.path.split(f)[1].split('.')[0]: f
                for f in glob.glob(os.path.join(dirname,
                                    '../ValidationMetrics/*.nc'))}

# matplotlib is optional dependency, only plot if installed - sjs 2022.01.28
if USE_MPL:
    ## Set display and meta parameters
    y1 = 1979
    y2 = 2016
    months = np.linspace(y1,y2+1,12*(y2-y1+1)+1)
    months = (months[:-1] + months[1:]) / 2
    years = np.arange(y1,y2+1)+0.5
    red_color     = (1,0.3,0.4)
    orange_color  = (255/256,140/256,0) 
    blue_color    = (0,0.447,0.741)
    purple_color  = (0.494,0.184,0.556)
    green_color   = (0.466,0.674,0.188)
    lightblue_color = (0.301,0.745,0.933)
    maroon_color  = (0.635,0.078,0.184)

    # add ability to save figures for validating new versions
    if SAVE_FIGS:
        matplotlib.use('Agg')
        figdir = os.path.join(dirname,'../ValidationMetrics/figs/')
        os.makedirs(figdir,exist_ok=True)

    ## 1) PSI -- Streamfunction zero crossing
    # read meridional velocity V(time,lat,lev), latitude and level
    V, lat, lev = get_arrs_from_nc(data_files['va'],['va','lat','lev'])
    #Change axes of V to be [time, lat, lev]
    V = np.transpose(V, (2,1,0))
    # Calculate metric from annual mean
    V_ANN = pyt.TropD_Calculate_Mon2Season(V, season=range(12))

    # latitude of monthly Psi zero crossing
    # Default method = 'Psi500'
    Phi_psi_sh, Phi_psi_nh = pyt.TropD_Metric_PSI(V, lat, lev)
    # latitude of NH/SH stream function zero crossing from annual mean V
    Phi_psi_sh_ANN, Phi_psi_nh_ANN = pyt.TropD_Metric_PSI(V_ANN, lat, lev)
    
    Phi_psi_nh_mean = pyt.TropD_Calculate_Mon2Season(Phi_psi_nh,
                                                     season=range(12))
    Phi_psi_sh_mean = pyt.TropD_Calculate_Mon2Season(Phi_psi_sh,
                                                     season=range(12))

    plt.figure(1,figsize=(7,7))
    plt.subplot(211)
    plt.plot(months, Phi_psi_nh, lw=1, c=green_color, 
             label='Lat of $\Psi_{500}$ zero crossing from monthly mean V')
    plt.plot(years, Phi_psi_nh_ANN, lw=2, c=blue_color,
             label='Lat of $\Psi_{500}$ zero crossing from annual mean V')
    plt.plot(years, Phi_psi_nh_mean, lw=2, c='k',
             label='Latitude of $\Psi_{500}$ zero crossing from annual '
                   'means of monthly metric values')
    plt.xticks(np.arange(1980,2020,5))
    plt.ylabel('latitude')
    plt.title(r"NH $\Psi_{500}$")
    plt.subplot(212)
    plt.plot(months, Phi_psi_sh, lw=1, c=green_color)
    plt.plot(years, Phi_psi_sh_ANN, lw=2, c=blue_color)
    plt.plot(years, Phi_psi_sh_mean, lw=2, c='k')
    plt.xticks(np.arange(1980,2020,5))
    plt.ylabel('latitude')
    plt.title(r"SH $\Psi_{500}$")
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'psi500_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()

    # Introduce latitude unertainty condition: no additional zero 
    # crossing is allowed within 10 degrees
    # latitude of monthly Psi zero crossing
    Phi_psi_sh_L, Phi_psi_nh_L = pyt.TropD_Metric_PSI(V, lat, lev,
                                                      method='Psi_500',
                                                      lat_uncertainty=10.)

    plt.figure(2)
    plt.plot(months,Phi_psi_nh,lw=2,c=green_color,
        label='$\Psi_{500}$ zero crossing from monthly mean V that '
              'meet uncertainty criterion') 
    plt.plot(months[np.isnan(Phi_psi_nh_L)],Phi_psi_nh[np.isnan(Phi_psi_nh_L)],
             marker='*', ls='None', markersize=10, c=red_color,
             label='$\Psi_{500}$ zero crossing from monthly mean V '
                   'that fail uncertainty criterion')
    plt.title(r'NH $\Psi_{500}$')
    plt.ylabel('latitude')
    plt.xlabel('Year')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.25, top=0.9, right=0.95)
    if SAVE_FIGS:
        figfile=os.path.join(figdir,f'psi500_uncertain_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()


    ## 2) TPB -- Tropopause break latitude
    # read temperature T(time,lat,lev), geopotential height [m] Z, lat and lev
    T, lat, lev = get_arrs_from_nc(data_files['ta'],['ta', 'lat', 'lev'])
    Z = get_arrs_from_nc(data_files['zg'],'zg')
    #Change axes of T and Z to be [time, lat, lev]
    T = np.transpose(T, (2,1,0))
    Z = np.transpose(Z, (2,1,0))
    # Calculate tropopause break from annual mean
    T_ANN = pyt.TropD_Calculate_Mon2Season(T, season=range(12))
    Z_ANN = pyt.TropD_Calculate_Mon2Season(Z, season=range(12))

    # Default method = "max_gradient". Latitude of maximal poleward gradient
    Phi_tpb_sh, Phi_tpb_nh = pyt.TropD_Metric_TPB(T, lat, lev)
    # (Default) latitude of maximal poleward gradient
    Phi_tpb_sh_ANN, Phi_tpb_nh_ANN = pyt.TropD_Metric_TPB(T_ANN, lat, lev,
                                                         method='max_gradient')
    # latitude of maximal difference in potential temperature
    # between the tropopase and surface 
    Phi_tpbT_sh_ANN, Phi_tpbT_nh_ANN = pyt.TropD_Metric_TPB(T_ANN, lat, lev,
                                                           method='max_potemp')
    # CutoffHeight = 15km marks the height of the tropopause break
    Phi_tpbZ_sh_ANN, Phi_tpbZ_nh_ANN = pyt.TropD_Metric_TPB(T_ANN, lat, lev, 
                                                            Z=Z_ANN,
                                                            method='cutoff')
    
    Phi_tpb_nh_mean = pyt.TropD_Calculate_Mon2Season(Phi_tpb_nh,
                                                     season=range(12))
    Phi_tpb_sh_mean = pyt.TropD_Calculate_Mon2Season(Phi_tpb_sh,
                                                     season=range(12))
    
    plt.figure(3,figsize=(7,7))
    plt.subplot(211)
    plt.plot(months, Phi_tpb_nh, lw=1, c=green_color,
             label='Lat. of tropopause break from monthly mean '
                   'T -- maximal gradient')
    plt.plot(years, Phi_tpb_nh_ANN, lw=2, c=blue_color,
             label='Lat. of tropopause break from annual mean '
                   'T -- maximal gradient')
    plt.plot(years, Phi_tpbZ_nh_ANN, ls='--', lw=1, c=blue_color,
             label='Lat. of tropopause break from annual mean '
                   'T -- 15km cutoff height')
    plt.plot(years, Phi_tpbT_nh_ANN, ls='--', lw=1, c=red_color,
             label='Lat. of tropopause break from annual mean '
                   'T -- pot. temp. difference')
    plt.plot(years, Phi_tpb_nh_mean, lw=2, c='k',
             label='Lat. of tropopause break from annual mean of '
                   'monthly metric -- pot. temp. difference')
    plt.title(r'NH tropopause break')
    plt.ylabel('latitude')
    plt.subplot(212)
    plt.plot(months, Phi_tpb_sh, lw=1, c=green_color)
    plt.plot(years, Phi_tpb_sh_ANN, lw=2, c=blue_color)
    plt.plot(years, Phi_tpbZ_sh_ANN, ls='--', lw=1, c=blue_color)
    plt.plot(years, Phi_tpbT_sh_ANN, ls='--', lw=1, c=red_color)
    plt.plot(years, Phi_tpb_sh_mean, lw=2, c='k')
    plt.xlabel('Year')
    plt.title(r'SH tropopause break')
    plt.ylabel('latitude')
    plt.gcf().legend(loc='lower center', frameon=False,fontsize=10)
    plt.subplots_adjust(bottom=0.25, hspace=0.3, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'trop_break_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()

    ##3) OLR -- OLR cutoff
    # Note: OLR is assumed to be positive upwards and in units of W/m^2
    # read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
    # read zonal mean monthly clear-sky TOA outgoing longwave radiation olrcs
    olr, lat = get_arrs_from_nc(data_files['rlnt'],['rlnt','lat'])
    olr *= -1.
    olrcs = -get_arrs_from_nc(data_files['rlntcs'],'rlntcs')
    #Change axes of olr and olrcs to be [time, lat]
    olr = np.transpose(olr, (1,0))
    olrcs = np.transpose(olrcs, (1,0))
    #Calculate annual mean field
    olr_ANN = pyt.TropD_Calculate_Mon2Season(olr, season=range(12))
    olrcs_ANN = pyt.TropD_Calculate_Mon2Season(olrcs, season=range(12))

    # latitude of olr metric
    # Default method = '250W'
    Phi_olr_sh, Phi_olr_nh = pyt.TropD_Metric_OLR(olr, lat)
    Phi_olrcs_sh, Phi_olrcs_nh = pyt.TropD_Metric_OLR(olrcs, lat)
    # latitude of olr metric from annual mean olr
    Phi_olr_sh_ANN, Phi_olr_nh_ANN = pyt.TropD_Metric_OLR(olr_ANN, lat)
    Phi_olrcs_sh_ANN, Phi_olrcs_nh_ANN = pyt.TropD_Metric_OLR(olrcs_ANN, lat)
    # latitude of olr metric from annual mean olr-20W/m^2 cutoff
    Phi_olr20_sh_ANN, Phi_olr20_nh_ANN = pyt.TropD_Metric_OLR(olr_ANN, lat,
                                                              method='20W')
    # latitude of olr metric from annual mean olr 240W/m^2 cutoff
    Phi_olr240_sh_ANN, Phi_olr240_nh_ANN = pyt.TropD_Metric_OLR(olr_ANN, lat,
                                                               method='cutoff',
                                                               Cutoff=240)

    plt.figure(4,figsize=(7,7))
    plt.subplot(211)
    plt.plot(months, Phi_olr_nh, lw=3, c=green_color,
             label='Lat of OLR 250W/m^2 cutoff (default) from monthly OLR')
    plt.plot(months, Phi_olrcs_nh, lw=1, c=tuple(0.5*x for x in green_color),
             label='Lat of OLR 250W/m^2 cutoff from monthly clear-sky OLR')
    plt.plot(years, Phi_olr_nh_ANN, lw=3, c=blue_color,
             label='Lat of OLR 250W/m^2 cutoff from annual mean OLR')
    plt.plot(years, Phi_olrcs_nh_ANN, lw=1, c=tuple(0.5*x for x in blue_color),
             label='Lat of OLR 250W/m^2 cutoff from annual mean clear-sky OLR')
    plt.ylabel('NH OLR cutoff latitude')
    plt.subplot(212)
    plt.plot(months, Phi_olr_sh, lw=3, c=green_color)
    plt.plot(months, Phi_olrcs_sh, lw=1, c=tuple(0.5*x for x in green_color))
    plt.plot(years, Phi_olr_sh_ANN, lw=3, c=blue_color)
    plt.plot(years, Phi_olrcs_sh_ANN, lw=1, c=tuple(0.5*x for x in blue_color))
    plt.xlabel('Year')
    plt.ylabel('SH OLR cutoff latitude')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'OLR_cutoff_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()

    plt.figure(5,figsize=(7,7))
    plt.subplot(211)
    plt.plot(years, Phi_olr_nh_ANN,lw=3, c=tuple(0.5*x for x in blue_color),
             label='Lat of OLR 250W/m^2 cutoff (default) from annual-mean OLR')
    plt.plot(years, Phi_olr240_nh_ANN, lw=3, c=blue_color,
             label='Lat of OLR 240W/m^2 cutoff from annual-mean OLR')
    plt.plot(years, Phi_olr20_nh_ANN, lw=3, c=green_color,
             label='Lat of OLR -20W/m^2 cutoff from annual-mean OLR')
    plt.ylabel('NH OLR cutoff latitude')
    plt.subplot(212)
    plt.plot(years, Phi_olr_sh_ANN, lw=3, c=tuple(0.5*x for x in blue_color))
    plt.plot(years, Phi_olr240_sh_ANN, lw=3, c=blue_color)
    plt.plot(years, Phi_olr20_sh_ANN, lw=3, c=green_color)
    plt.xlabel('Year')
    plt.ylabel('SH OLR cutoff latitude')
    plt.ylabel('latitude')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'OLR_vary_cutoff_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()


    ## 4) STJ -- Subtropical Jet (STJ) latitude
    #read zonal wind U(time,lat,lev), latitude and level
    U, lat, lev = get_arrs_from_nc(data_files['ua'],['ua','lat','lev'])

    #Change axes of u to be [time, lat]
    U = np.transpose(U, (2,1,0))

    # Calculate STJ latitude from annual mean
    U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=range(12))
    # latitude of STJ from annual mean U
    # Default method =  'adjusted_peak'
    Phi_stj_sh_ANN_adj, Phi_stj_nh_ANN_adj = pyt.TropD_Metric_STJ(
                                                        U_ANN, 
                                                        lat,
                                                        lev,
                                                        method='adjusted_peak')
    Phi_stj_sh_ANN_core, Phi_stj_nh_ANN_core = pyt.TropD_Metric_STJ(
                                                        U_ANN,
                                                        lat,
                                                        lev,
                                                        method='core_peak')

    plt.figure(6,figsize=(7,7))
    plt.subplot(211)
    plt.plot(years, Phi_stj_nh_ANN_adj, lw=2, c=green_color,
             label='Lat of STJ from anual mean U, using "adjusted peak"')
    plt.plot(years, Phi_stj_nh_ANN_core, lw=2, c=blue_color,
             label='Lat of STJ from anual mean U, using "core peak"')
    plt.ylabel('NH STJ latitude')
    plt.subplot(212)
    plt.plot(years, Phi_stj_sh_ANN_adj, lw=2, c=green_color)
    plt.plot(years, Phi_stj_sh_ANN_core, lw=2, c=blue_color)
    plt.xlabel('Year')
    plt.ylabel('SH STJ latitude')
    plt.ylabel('latitude')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.15, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'STJ_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()

    ## 5) EDJ -- Eddy Driven Jet (EDJ) latitude
    #read zonal wind U(time,lat,lev), latitude and level
    U, lat, lev = get_arrs_from_nc(data_files['ua'],['ua','lat','lev'])
    #Change axes of u to be [time, lat]
    U = np.transpose(U, (2,1,0))
    Phi_edj_sh, Phi_edj_nh = pyt.TropD_Metric_EDJ(U, lat, lev, method='max')

    # Calculate EDJ latitude from annual mean
    U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=range(12))
    # latitude of EDJ from annual mean U
    Phi_edj_sh_ANN, Phi_edj_nh_ANN = pyt.TropD_Metric_EDJ(U_ANN, lat, lev)
    
    Phi_edj_nh_mean = pyt.TropD_Calculate_Mon2Season(Phi_edj_nh,
                                                     season=range(12))
    Phi_edj_sh_mean = pyt.TropD_Calculate_Mon2Season(Phi_edj_sh,
                                                     season=range(12))

    plt.figure(7,figsize=(7,7))
    plt.subplot(211)
    plt.plot(months, Phi_edj_nh, lw=1, c=green_color,
             label='Lat of EDJ from monthly mean U')
    plt.plot(years, Phi_edj_nh_ANN, lw=2, c=blue_color,
             label='Lat of EDJ from annual mean U')
    plt.plot(years, Phi_edj_nh_mean, lw=2, c='k',
             label='Lat of EDJ from annual mean of monthly metric values')
    plt.ylabel('NH EDJ latitude')
    plt.subplot(212)
    plt.plot(months, Phi_edj_sh, lw=1, c=green_color)
    plt.plot(years, Phi_edj_sh_ANN, lw=2, c=blue_color)
    plt.plot(years, Phi_edj_sh_mean, lw=2, c='k')
    plt.xlabel('Year')
    plt.ylabel('SH EDJ latitude')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'EDJ_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()


    ## 6) PE -- Precipitation minus evaporation subtropical zero crossing lat
    # read zonal mean monthly precipitation pr(time,lat)
    pr, lat = get_arrs_from_nc(data_files['pr'],['pr','lat'])
    # read zonal mean monthly evaporation drived from
    # surface latent heat flux hfls(time,lat)
    er = -get_arrs_from_nc(data_files['hfls'],'hfls')/L_VAP
    PE = pr - er 

    #Change axes of pr and er to be [time, lat]
    PE = np.transpose(PE, (1,0))

    PE_ANN = pyt.TropD_Calculate_Mon2Season(PE, season=range(12))

    # latitude of PminusE metric
    # Default method = 'zero_crossing'
    Phi_pe_sh, Phi_pe_nh = pyt.TropD_Metric_PE(PE, lat)
    # latitude of PminusE metric from annual mean PminusE
    Phi_pe_sh_ANN, Phi_pe_nh_ANN = pyt.TropD_Metric_PE(PE_ANN, lat)
    
    Phi_pe_nh_mean = pyt.TropD_Calculate_Mon2Season(Phi_pe_nh,season=range(12))
    Phi_pe_sh_mean = pyt.TropD_Calculate_Mon2Season(Phi_pe_sh,season=range(12))

    plt.figure(8,figsize=(7,7))
    plt.subplot(211)
    plt.plot(months, Phi_pe_nh, lw=2, c=green_color,
             label='Lat of P$-$E zero-crossing')
    plt.plot(years, Phi_pe_nh_ANN, lw=2, c=blue_color,
             label='Lat of P$-$E zero-crossing from annual mean field')
    plt.plot(years, Phi_pe_nh_mean, lw=2, c='k',
             label='Lat of P$-$E zero-crossing from annual '
                   'mean of monthly metric')
    plt.ylabel('NH P$-$E zero-crossing')
    plt.subplot(212)
    plt.plot(months, Phi_pe_sh, lw=2, c=green_color)
    plt.plot(years, Phi_pe_sh_ANN, lw=2, c=blue_color)
    plt.plot(years, Phi_pe_sh_mean, lw=2, c='k')
    plt.xlabel('Year')
    plt.ylabel('SH P$-$E zero-crossing')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'P-E_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()


    ## 7) UAS -- Zonal surface wind subtropical zero crossing latitude
    #read zonal wind U(time,lat,lev), latitude and level
    U, lat, lev = get_arrs_from_nc(data_files['ua'],['ua','lat','lev'])
    #read zonal mean surface wind U(time,lat)
    uas = get_arrs_from_nc(data_files['uas'],'uas')

    #Change axes of u to be [time, lat]
    U = np.transpose(U, (2,1,0))
    uas = np.transpose(uas, (1,0))

    uas_ANN = pyt.TropD_Calculate_Mon2Season(uas, season=range(12))
    U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=range(12))

    # latitude of surface zonal wind metric
    Phi_uas_sh, Phi_uas_nh = pyt.TropD_Metric_UAS(uas, lat)
    # latitude of surface zonal wind metric from annual mean surface zonal wind
    Phi_uas_sh_ANN, Phi_uas_nh_ANN = pyt.TropD_Metric_UAS(uas_ANN, lat)
    # latitude of surface zonal wind metric from annual mean zonal wind
    Phi_Uas_sh_ANN, Phi_Uas_nh_ANN = pyt.TropD_Metric_UAS(U_ANN, lat, lev)
    
    Phi_uas_nh_mean = pyt.TropD_Calculate_Mon2Season(Phi_uas_nh, 
                                                     season=range(12))
    Phi_uas_sh_mean = pyt.TropD_Calculate_Mon2Season(Phi_uas_sh,
                                                     season=range(12))

    plt.figure(9,figsize=(7,7))
    plt.subplot(211)
    plt.plot(months, Phi_uas_nh, lw=2, c=green_color,
        label='Lat of surface zonal wind zero crossing')
    plt.plot(years, Phi_uas_nh_ANN, lw=2, c=blue_color,
        label='Lat of surface zonal wind zero crossing from annual mean field')
    plt.plot(years, Phi_Uas_nh_ANN, lw=2, c=red_color,
        label='Lat of 850 hPa zonal wind zero crossing from annual mean field')
    plt.plot(years, Phi_uas_nh_mean, lw=2, c='k',
        label='Lat of surface zonal wind zero crossing from'
              ' annual mean of monthly metric')
    plt.ylabel('NH uas zero-crossing')
    plt.subplot(212)
    plt.plot(months, Phi_uas_sh, lw=2, c=green_color)
    plt.plot(years, Phi_uas_sh_ANN, lw=2, c=blue_color)
    plt.plot(years, Phi_Uas_sh_ANN, lw=2, c=red_color)
    plt.plot(years, Phi_uas_sh_mean, lw=2, c='k',)
    plt.xlabel('Year')
    plt.ylabel('SH uas zero-crossing')
    plt.gcf().legend(loc='lower center', frameon=False)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, top=0.95, right=0.95)
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'usfc_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()

    ## 8) PSL -- Sea-level Pressure Maximum
    # read sea-level pressure ps(time,lat) and latitude
    ps, lat = get_arrs_from_nc(data_files['psl'],['psl','lat'])

    #Change axes of ps to be [time, lat]
    ps = np.transpose(ps, (1,0))

    # calculate DJF/JJA means
    ps_DJF = pyt.TropD_Calculate_Mon2Season(ps, season=[0,1,11]) 
    ps_JJA = pyt.TropD_Calculate_Mon2Season(ps, season=[5,6,7])

    # latitude of monthly max surface pressure
    # Default method = 'peak'
    Phi_ps_DJF_sh, Phi_ps_DJF_nh = pyt.TropD_Metric_PSL(ps_DJF, lat)
    Phi_ps_JJA_sh, Phi_ps_JJA_nh = pyt.TropD_Metric_PSL(ps_JJA, lat)

    plt.figure(10)
    plt.subplot(211)
    plt.plot(years, Phi_ps_DJF_nh, lw=2, c=green_color,
             label='Lat of max sea-level pressure during DJF')
    plt.plot(years, Phi_ps_JJA_nh, lw=2, c=blue_color,
             label='Lat of max sea-level pressure during JJA')
    plt.ylabel('NH max psl latitude')
    plt.legend(loc='best', frameon=False)
    plt.subplot(212)
    plt.plot(years, Phi_ps_DJF_sh, lw=2, c=green_color)
    plt.plot(years, Phi_ps_JJA_sh, lw=2, c=blue_color)
    plt.ylabel('SH max psl latitude')
    plt.tight_layout()
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'PSL_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()

    ## 9) Compare annual mean metrics
    # Psi500
    # read meridional velocity V(time,lat,lev), latitude and level
    V, lat, lev = get_arrs_from_nc(data_files['va'],['va','lat','lev'])
    # Change axes of V to be [time, lat]
    V = np.transpose(V, (2,1,0))
    V_ANN = pyt.TropD_Calculate_Mon2Season(V, season=range(12))
    Phi_psi_sh_ANN, Phi_psi_nh_ANN = pyt.TropD_Metric_PSI(V_ANN, lat, lev)

    # Tropopause break
    # read meridional temperature T(time,lat,lev), latitude and level
    T = get_arrs_from_nc(data_files['ta'],'ta')
    # Change axes of T to be [time, lat]
    T = np.transpose(T, (2,1,0))
    T_ANN = pyt.TropD_Calculate_Mon2Season(T, season=range(12))
    Phi_tpb_sh_ANN, Phi_tpb_nh_ANN = pyt.TropD_Metric_TPB(T_ANN, lat, lev)

    # Surface pressure max
    # read sea-level pressure ps(time,lat) and latitude
    ps = get_arrs_from_nc(data_files['psl'],'psl')
    # Change axes of ps to be [time, lat]
    ps = np.transpose(ps, (1,0))
    ps_ANN = pyt.TropD_Calculate_Mon2Season(ps, season=range(12))
    Phi_ps_sh_ANN, Phi_ps_nh_ANN = pyt.TropD_Metric_PSL(ps_ANN, lat)

    # Eddy driven jet
    # read zonal wind U(time,lat,lev), latitude and level
    U = get_arrs_from_nc(data_files['ua'],'ua')
    # Change axes of U to be [time, lat]
    U = np.transpose(U, (2,1,0))
    U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=range(12))
    Phi_edj_sh_ANN, Phi_edj_nh_ANN = pyt.TropD_Metric_EDJ(U_ANN, lat, lev)

    # Subtropical jet
    Phi_stj_sh_ANN, Phi_stj_nh_ANN = pyt.TropD_Metric_STJ(U_ANN, lat, lev)

    # OLR
    # read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
    olr = -get_arrs_from_nc(data_files['rlnt'],'rlnt')
    # Change axes of olr to be [time, lat]
    olr = np.transpose(olr, (1,0))
    olr_ANN = pyt.TropD_Calculate_Mon2Season(olr, season=range(12))
    Phi_olr_sh_ANN, Phi_olr_nh_ANN = pyt.TropD_Metric_OLR(olr_ANN, lat)

    # P minus E
    # read zonal mean monthly precipitation pr(time,lat)
    pr = get_arrs_from_nc(data_files['pr'],'pr')
    # read zonal mean monthly evaporation derived from 
    # surface latent heat flux hfls(time,lat)
    er = -get_arrs_from_nc(data_files['hfls'],'hfls')/L_VAP
    PE = pr - er
    # Change axes of PE to be [time, lat]
    PE = np.transpose(PE, (1,0))
    PE_ANN = pyt.TropD_Calculate_Mon2Season(PE, season=range(12))
    Phi_pe_sh_ANN, Phi_pe_nh_ANN = pyt.TropD_Metric_PE(PE_ANN, lat)

    # Surface winds
    # read zonal mean surface wind U(time,lat)
    uas = get_arrs_from_nc(data_files['uas'],'uas')
    # Change axes of uas to be [time, lat]
    uas = np.transpose(uas, (1,0))
    uas_ANN = pyt.TropD_Calculate_Mon2Season(uas, season=range(12))
    Phi_uas_sh_ANN, Phi_uas_nh_ANN = pyt.TropD_Metric_UAS(uas_ANN, lat)


    plt.figure(11)
    plt.subplot(211)
    plt.plot(years, Phi_psi_nh_ANN, lw=2, c=(0,0,0), label='PSI')
    plt.plot(years, Phi_tpb_nh_ANN, lw=2, c=green_color, label='TPB')
    plt.plot(years, Phi_edj_nh_ANN, lw=2, c=blue_color, label='EDJ')
    plt.plot(years, Phi_stj_nh_ANN, lw=2, c=red_color, label='STJ')
    plt.plot(years, Phi_olr_nh_ANN, lw=2, c=lightblue_color, label='OLR')
    plt.plot(years, Phi_pe_nh_ANN, lw=2, c=orange_color, label='P-E')
    plt.plot(years, Phi_uas_nh_ANN, lw=2, c=purple_color, label='UAS')
    plt.plot(years, Phi_ps_nh_ANN, ls='--', lw=2, c=maroon_color, label='PSL')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.ylabel('NH HC edge')
    
    plt.subplot(212)
    plt.plot(years, Phi_psi_sh_ANN, lw=2, c=(0,0,0), label='PSI')
    plt.plot(years, Phi_tpb_sh_ANN, lw=2, c=green_color, label='TPB')
    plt.plot(years, Phi_edj_sh_ANN, lw=2, c=blue_color, label='EDJ')
    plt.plot(years, Phi_stj_sh_ANN, lw=2, c=red_color, label='STJ')
    plt.plot(years, Phi_olr_sh_ANN, lw=2, c=lightblue_color, label='OLR')
    plt.plot(years, Phi_pe_sh_ANN, lw=2, c=orange_color, label='P-E')
    plt.plot(years, Phi_uas_sh_ANN, lw=2, c=purple_color, label='UAS')
    plt.plot(years, Phi_ps_sh_ANN, ls='--', lw=2, c=maroon_color, label='PSL')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.xlabel('Year')
    plt.ylabel('SH HC edge')
    plt.tight_layout()
    if SAVE_FIGS:
        figfile = os.path.join(figdir,f'all_ann_metrics_{pyt.__version__}.png')
        plt.savefig(figfile)
        plt.close()
        print('ValidationMetrics/figs/'+os.path.split(figfile)[1]+' saved!')
    else:
        plt.show()


## 10) Validate metrics
checks_passed = 0
# Check calculations with precalculated values from file within roundoff error
#Psi500
#read meridional velocity V(time,lat,lev), latitude and level
V, lat, lev = get_arrs_from_nc(data_files['va'],['va','lat','lev'])
#Change axes of V to be [time, lat]
V = np.transpose(V, (2,1,0))
#resample to annual
V_ANN = pyt.TropD_Calculate_Mon2Season(V, season=range(12))

#monthly PSI check
Phi_psi_sh, Phi_psi_nh = pyt.TropD_Metric_PSI(V, lat, lev)
checks_passed += validate_against_metric(Phi_psi_nh, 'PSI', 'NH',
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_psi_sh, 'PSI', 'SH',
                                         verbose=VERBOSE)

#annual PSI check
Phi_psi_sh_ANN, Phi_psi_nh_ANN = pyt.TropD_Metric_PSI(V_ANN, lat, lev)
checks_passed += validate_against_metric(Phi_psi_nh_ANN, 'PSI_ANN', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_psi_sh_ANN, 'PSI_ANN', 'SH', 
                                         verbose=VERBOSE)


# Tropopause break
#read temperature T(time,lat,lev), latitude and level
T = get_arrs_from_nc(data_files['ta'],'ta')
#Change axes of T to be [time, lat, lev]
T = np.transpose(T, (2,1,0))
#resample to annual
T_ANN = pyt.TropD_Calculate_Mon2Season(T, season=range(12))

#monthly TPB check
Phi_tpb_sh, Phi_tpb_nh = pyt.TropD_Metric_TPB(T, lat, lev)
checks_passed += validate_against_metric(Phi_tpb_nh, 'TPB', 'NH',
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_tpb_sh, 'TPB', 'SH',
                                         verbose=VERBOSE)

#annual TPB check
Phi_tpb_sh_ANN, Phi_tpb_nh_ANN = pyt.TropD_Metric_TPB(T_ANN, lat, lev)
checks_passed += validate_against_metric(Phi_tpb_nh_ANN, 'TPB_ANN', 'NH',
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_tpb_sh_ANN, 'TPB_ANN', 'SH',
                                         verbose=VERBOSE)


# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
ps = get_arrs_from_nc(data_files['psl'],'psl')
#Change axes of ps to be [time, lat]
ps = np.transpose(ps, (1,0))
#loop over seasons
for season in ['DJF','MAM','JJA','SON']:
    #dict linking months to resample with string season code
    #to enable looping
    season_array = {'DJF':[0,1,11], 'MAM':[2,3,4],
                    'JJA':[5,6,7],  'SON':[8,9,10]}[season]
    
    #resample to seasonal
    ps_seasonal = pyt.TropD_Calculate_Mon2Season(ps, season=season_array)
    Phi_ps_sh_ssn, Phi_ps_nh_ssn = pyt.TropD_Metric_PSL(ps_seasonal, lat)
    checks_passed += validate_against_metric(Phi_ps_nh_ssn, 'PSL_'+season,'NH',
                                             verbose=VERBOSE)
    checks_passed += validate_against_metric(Phi_ps_sh_ssn, 'PSL_'+season,'SH',
                                             verbose=VERBOSE)


# Eddy driven jet
#read zonal wind U(time,lat,lev), latitude and level
U = get_arrs_from_nc(data_files['ua'],'ua')
#Change axes of U to be [time, lat]
U = np.transpose(U, (2,1,0))
#resample to annual
U_ANN = pyt.TropD_Calculate_Mon2Season(U, season=range(12))

#monthly EDJ check
Phi_edj_sh, Phi_edj_nh = pyt.TropD_Metric_EDJ(U, lat, lev)
checks_passed += validate_against_metric(Phi_edj_nh, 'EDJ', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_edj_sh, 'EDJ', 'SH', 
                                         verbose=VERBOSE)

#annual EDJ check
Phi_edj_sh_ANN, Phi_edj_nh_ANN = pyt.TropD_Metric_EDJ(U_ANN, lat, lev)
checks_passed += validate_against_metric(Phi_edj_nh_ANN, 'EDJ_ANN', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_edj_sh_ANN, 'EDJ_ANN', 'SH', 
                                         verbose=VERBOSE)


# Subtropical jet
#monthly STJ check
Phi_stj_sh, Phi_stj_nh = pyt.TropD_Metric_STJ(U, lat, lev)
checks_passed += validate_against_metric(Phi_stj_nh, 'STJ', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_stj_sh, 'STJ', 'SH', 
                                         verbose=VERBOSE)

#annual STJ check
Phi_stj_sh_ANN, Phi_stj_nh_ANN = pyt.TropD_Metric_STJ(U_ANN, lat, lev)
checks_passed += validate_against_metric(Phi_stj_nh_ANN, 'STJ_ANN', 'NH',
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_stj_sh_ANN, 'STJ_ANN', 'SH', 
                                         verbose=VERBOSE)


# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
olr = -get_arrs_from_nc(data_files['rlnt'],'rlnt')
#Change axes of olr to be [time, lat]
olr = np.transpose(olr, (1,0))
#resample to annual
olr_ANN = pyt.TropD_Calculate_Mon2Season(olr, season=range(12))

#monthly OLR check
Phi_olr_sh, Phi_olr_nh = pyt.TropD_Metric_OLR(olr, lat)
checks_passed += validate_against_metric(Phi_olr_nh, 'OLR', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_olr_sh, 'OLR', 'SH', 
                                         verbose=VERBOSE)

#annual OLR check
Phi_olr_sh_ANN, Phi_olr_nh_ANN = pyt.TropD_Metric_OLR(olr_ANN, lat)
checks_passed += validate_against_metric(Phi_olr_nh_ANN, 'OLR_ANN', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_olr_sh_ANN, 'OLR_ANN', 'SH', 
                                         verbose=VERBOSE)


# P minus E
# read zonal mean monthly precipitation pr(time,lat)
pr = get_arrs_from_nc(data_files['pr'],'pr')
# read zonal mean monthly evaporation drived from 
# surface latent heat flux hfls(time,lat)
er = -get_arrs_from_nc(data_files['hfls'],'hfls')/L_VAP
PE = pr - er
#Change axes of PE to be [time, lat]
PE = np.transpose(PE, (1,0))
#resample to annual
PE_ANN = pyt.TropD_Calculate_Mon2Season(PE, season=range(12))

#monthly P-E check
Phi_pe_sh, Phi_pe_nh = pyt.TropD_Metric_PE(PE, lat)
checks_passed += validate_against_metric(Phi_pe_nh, 'PE', 'NH',
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_pe_sh, 'PE', 'SH',
                                         verbose=VERBOSE)

#annual P-E check
Phi_pe_sh_ANN, Phi_pe_nh_ANN = pyt.TropD_Metric_PE(PE_ANN, lat)
checks_passed += validate_against_metric(Phi_pe_nh_ANN, 'PE_ANN', 'NH',
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_pe_sh_ANN, 'PE_ANN', 'SH',
                                         verbose=VERBOSE)


# Surface winds
#read zonal mean surface wind U(time,lat)
uas = get_arrs_from_nc(data_files['uas'],'uas')
#Change axes of uas to be [time, lat]
uas = np.transpose(uas, (1,0))
#resample to annual
uas_ANN = pyt.TropD_Calculate_Mon2Season(uas, season=range(12))

# monthly UAS check
Phi_uas_sh, Phi_uas_nh = pyt.TropD_Metric_UAS(uas, lat)
checks_passed += validate_against_metric(Phi_uas_nh, 'UAS', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_uas_sh, 'UAS', 'SH', 
                                         verbose=VERBOSE)

# annual UAS check
Phi_uas_sh_ANN, Phi_uas_nh_ANN = pyt.TropD_Metric_UAS(uas_ANN, lat)
checks_passed += validate_against_metric(Phi_uas_nh_ANN, 'UAS_ANN', 'NH', 
                                         verbose=VERBOSE)
checks_passed += validate_against_metric(Phi_uas_sh_ANN, 'UAS_ANN', 'SH', 
                                         verbose=VERBOSE)

print(f'{checks_passed} out of 36 metric checks validated!')