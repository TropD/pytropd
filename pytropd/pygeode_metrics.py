from pygeode.var import Var
from pygeode.dataset import Dataset
import numpy as np
#import pytropd as pyt
import metrics as pyt

import pygeode as pyg
from pygeode.axis import NamedAxis
from pygeode.tools import loopover, whichaxis 
from pygeode.view import View
from scipy import interpolate
from inspect import signature
import logging

def metrics_dataset(dataset, metric, **params):
    metric_property_name = dict(edj=['u', 'Zonal wind'],
                                olr=['olr','Outgoing longwave radiation'],
                                pe=['pe','Precipitation minus evaporation'],
                                psi=['v','Meridional wind'],
                                psl=['psl','Sea level pressure'],
                                stj=['u','Zonal wind'],
                                tpb=['T','Temperature'],
                                uas=['uas','Surface wind'],
                                gwl=['tracer','Tracer'],
                                )


    p_axis_status = pressure_axis_status(metric)
    
    # make sure data is a dataset not a var
    if isinstance(dataset, pyg.Var): 
      dataset = pyg.asdataset(dataset)

    dataset = extract_property_name(dataset, property_name=['lat','Latitude'], p_axis_status=p_axis_status)
    dataset = extract_property_name(dataset, property_name=['pres','Pressure'], p_axis_status=p_axis_status)
    #if dataset only contains one pyg.Var, assume
    #this is the correct variable for the metric and return it
    #otherwise, find check the name of the variable for the closest match
    # If none found, raise error.
    var = extract_property_name(dataset, property_name=metric_property_name[metric], p_axis_status=p_axis_status)

    print('nh')
    #nh hemisphere
    nh_data = chop_by_hemisphere(var,hem='nh')
    nh_Var = metric_var(nh_data, output='lat', p_axis_status=p_axis_status, metric=metric, hem='nh', **params)
    
    
    print('sh')
    #sh hemisphere
    sh_data = chop_by_hemisphere(var, hem='sh')
    sh_Var = metric_var(sh_data, output='lat', p_axis_status=p_axis_status, metric=metric, hem='sh', **params)
   
    return pyg.Dataset([sh_Var, nh_Var])


def metric_var(X, output='lat', p_axis_status=None, metric=None, hem='nh', pbar=None, **params):
  ''''''
  
  # Get the relevant metric function
  metric_function = getattr(pyt, 'TropD_Metric_' + metric.upper()) 

  # if method not provided, get default method of metric_function
  method_used: str = params.get(
       "method", signature(metric_function).parameters["method"].default)
  
  p_axis_status=0
  ovars = ['lat','values']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from metric calculation. Possible outputs are %s.' % str(ovars))

  xn = X.name if X.name != '' else 'X' # Note: could write:  xn = X.name or 'X'

  inaxes = list(X.axes)
  out_order = [type(i) for i in inaxes]
  if X.hasaxis('Lat'):
    has_pres_axis = 0
    
    #move lat axis to the end
    out_order.append(out_order.pop(X.whichaxis('Lat')))

    if X.hasaxis('Pres') and (p_axis_status==1):
      #move pres axis to the end
      out_order.append(out_order.pop(X.whichaxis('Pres')))
      has_pres_axis = 1

    else:
      if p_axis_status==1:
        print(X)
        raise KeyError('<Pres> axis not found in', X)
  
  else: 
    print(X)
    raise KeyError('<Lat> axis not found in', X)

  
  #transpose axes in the order (Lat, Pres)
  X = X.transpose(*out_order)
  lataxis = X.whichaxis('Lat')
  lat_values = X.axes[lataxis][:]

  riaxes = [lataxis]
  if p_axis_status==1:
    presaxis = X.whichaxis('Pres')
    lev = X.axes[presaxis][:]
    params['lev'] = lev
    riaxes.append(presaxis)

  
  oaxes = [a for i, a in enumerate(X.axes) if i not in riaxes]
  print(oaxes) 
  if not oaxes:
    new_const_axis = NamedAxis(values=np.arange(1),name='value')
    X = X.extend(0, new_const_axis)
    oaxes.append(new_const_axis)
    inaxes = list(X.axes)

  raxes = [a for i, a in enumerate(X.axes) if i in riaxes]

  # Construct new variable
  if hem == 'nh':
    name = metric.upper() + '_nh_latitude'
  else: 
    name = metric.upper() + '_sh_latitude'

  oview = View(oaxes) 
  iview = View(raxes) 

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar()

  outview = View(oaxes)

  # Construct work arrays
  metric_lat= np.full(oview.shape, np.nan, 'd')
  metric_value = np.full(oview.shape, np.nan, 'd')



  ## Vectorize the metric function depending on inputs needed
  #if has_pres_axis:
  #  signature = '(i,j)->(k)'
  #else:
  #  signature = '(i)->(k)'
  ##metric_function_v = np.vectorize(metric_function, signature=signature)
  
  
  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, inaxes, pbar=pbar):
    xdata = xdata.astype('d')

    if method_used =='fit':
      metric_lat[outsl], metric_value[outsl] = metric_function(xdata, lat_values, **params)
    else:
      metric_lat[outsl] = metric_function(xdata, lat_values, **params)

  var_list_out = []

  if 'lat' in output:
    lat_attrs = {"long_name": metric.upper() + " metric latitude",
                 "unit": "degrees"},
    metric_lat = Var(oaxes, values=metric_lat, name=hem + '_metric_lat', atts=lat_attrs)
    var_list_out.append(metric_lat)

  ds = asdataset(var_list_out)
  pbar.update(100)
  return ds



def pressure_axis_status(metric):
  # metrics that take 1D variable as input
  if metric in ['edj','olr','pe','psl','uas']:
    return 0

  #metrics that takes a 2D variable as input
  elif metric in ['psi','stj','tpb']:
    return 1
  
  #metrics that can take a 2D variable as input but do not collapse pressure axis
  elif metric in ['gwl', 'onesigma']:
    return 2

def extract_property_name(dataset, property_name, p_axis_status): 

  find_index = 0

  if property_name[0] == 'lat':
    #check if pyg.Lat axis is present, else look for a match
    if dataset.hasaxis(pyg.Lat):
      print('Found Latitude axis in the dataset')
      return dataset

    else:
      dataset_keys = list(dataset.axisdict.keys())
      property_name_list = ['lat','latitude','lats','x','phi','degreesnorth']
      find_index = 1

  elif property_name[0] == 'pres':  
    #check if pyg.Pres axis is present, else look for a match
    if dataset.hasaxis(pyg.Pres):
      print('Found Pressure axis in the dataset')
      return dataset

    else:
      #try to see if there is a pressure axis as a pyg.NamedAxis
      dataset_keys = list(dataset.axisdict.keys())
      property_name_list = ['pres','pressure','p','lev','levels','level']
      find_index = 1

  else:
    dataset_keys = list(dataset.vardict.keys())

    #if we are only given one data array in the dataset, assume it is the right one,
    if len(dataset_keys) == 1:
      index = [0]

    #otherwise look for a matching variable name 
    else:
      if property_name[0] == 'u':
        property_name_list = ['zonalwind','uwind','u','xwind']
        if property_name[0] == 'uas':
          property_name_list = ['surfacewind','uas','us','surfu','usurf']
        elif property_name[0] == 'v':
          property_name_list = ['meridionalwind','vwind','v','ywind']
        elif property_name[0] == 'T':
          property_name_list = ['t','temp','temperature']
        elif property_name[0] == 'psl':
          property_name_list = ['sealevelpressure','slp','psl','ps','sp']
        elif property_name[0] == 'olr':
          property_name_list = ['olr','outgoinglongwaveradiation','toaolr','olrtoa']
        elif property_name[0] == 'pe':
          property_name_list = ['pe','precipitationminusevarporation','pminuse']
        find_index = 1

  if find_index:
    #array names in dataset. Remove whitespace, underscores and make lowercase
    array_names = [string.strip().lower().replace('_','').replace('-','') for string in dataset_keys]
    #create dict of indices in dataset
    indices_dict = dict((k,i) for i,k in enumerate(array_names)) 
    #find variable
    intersection = set(indices_dict).intersection(property_name_list)  
    #extract relevant index
    index = [indices_dict[x] for x in intersection]

  if property_name[0] == 'pres':
    #Return an error if we need a pressure axis
    if p_axis_status and len(index)==0: 
      print(dataset)
      raise KeyError('%s not found in Dataset. Valid variable names are %s'%(property_name[1],property_name_list))

    elif len(index)==1:
      #if we find a pres axis convert it to a pyg.Pres axis and return data
      print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
      pres_axis = getattr(dataset, dataset_keys[index[0]])
      print('Replacing pyg.NamedAxis %s with a pyg.Pres axis in the dataset'%(dataset_keys[index[0]]))
      return dataset.replace_axes({dataset_keys[index[0]]: pyg.Pres(pres_axis[:])})

    else:
      return dataset

  if len(index)==0: 
    print(dataset)
    raise KeyError('%s not found in Dataset. Valid variable names are %s'%(property_name[1],property_name_list))


  if len(index)>1:
    print(dataset)
    print('More than one possible key for %s found. Valid variable names are %s'%(property_name[1],property_name_list))
    raise KeyError

  if property_name[0] == 'lat' and find_index:
    print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
    lat_axis = getattr(dataset, dataset_keys[index[0]])
    print('Replacing pyg.NamedAxis %s with a pyg.Lat axis in the dataset'%(dataset_keys[index[0]]))
    return dataset.replace_axes({dataset_keys[index[0]]: pyg.Lat(lat_axis[:])})


  else:
    print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
    return getattr(dataset, dataset_keys[index[0]])
  
  
def chop_by_hemisphere(dataset, hem='nh'):
  if dataset.hasaxis('Lat'):
    minlat = dataset.lat.min() 
    maxlat = dataset.lat.max() 
    if (hem == 'nh' and maxlat > 0):
      return dataset(lat=(0,90))
    elif (hem == 'sh' and minlat < 0):
      return dataset(lat=(-90,0))
    else:
      return None

  else:
    raise KeyError('Latitude axis not found')

def pyg_edj(dataset,**params):

  '''TropD Eddy Driven Jet (EDJ) metric

     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     Var should contain axis :class:`pyg.Lat`. If :class:`pyg.Pres` is given, level closest to 850hPa is chosen 

     Parameters
        method (str, optional): 'peak' (default) |  'max' | 'fit'

        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)

        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum

       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

     Returns:
       EDJ_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)

  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> U = t2.U(i_time=0).mean(pyg.Lon)
  >>> print(U)
  <Var 'U'>:
    Shape:  (pres,lat)  (20,31)
  Axes:
    pres <Pres>    :  1000 hPa to 50 hPa (20 values)
    lat <Lat>      :  90 S to 90 N (31 values)
  Attributes:
    {}
  Type:  SqueezedVar (dtype="float64")
  >>> print(pyg_edj(U))
  <Var 'EDJ_metrics'>:
    Shape:  (Metrics)  (2)
  Axes:
    Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
  Attributes:
    {}
  Type:  Replace_axes (dtype="float64")
  >>> print(edj(U)[:])
  [-45.  45.]
  '''

  EDJ_Dataset = metrics_dataset(dataset, metric='edj', **params)

  return EDJ_Dataset


def pyg_olr(dataset, **params):

  """TropD Outgoing Longwave Radiation (OLR) metric
     
     Var should contain one axis :class:`pyg.Lat`.  
     Parameters:
     
       olr(lat,): zonal mean TOA olr (positive)
       
       lat: equally spaced latitude column vector
        
       method (str, optional):

         '250W'(Default): the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses 250W/m^2
         
         '20W': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses the tropical OLR max minus 20W/m^2
         
         'cutoff': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses a specified cutoff value
         
         '10Perc': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR is 10# smaller than the tropical OLR maximum
         
         'max': the latitude of maximum of tropical olr in each hemisphere with the smoothing paramerer n=6 in TropD_Calculate_MaxLat
         
         'peak': the latitude of maximum of tropical olr in each hemisphere with the smoothing parameter n=30 in TropD_Calculate_MaxLat
       
       
       Cutoff (float, optional): Scalar. For the method 'cutoff', Cutoff specifies the OLR cutoff value. 
       
       n (int, optional): For the 'max' method, n is the smoothing parameter in TropD_Calculate_MaxLat
     
     Returns:
       OLR_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)
  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> olr_data = 150*pyg.cosd(t2.lat*2) +150 #fake olr data
  >>> print(olr_data)                                                                      
  <Var '((cosd(lat)*150)+150)'>:
    Shape:  (lat)  (31)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
    Attributes:
      {}
    Type:  Add_Var (dtype="float64")
  >>> print(olr(olr_data)) #Calculate OLR metrics                                           
  <Var 'OLR_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  
  >>> print(olr(olr_data)[:])                                                              
  [-24.0874096  24.0874096]
  """

  OLR_Dataset = metrics_dataset(dataset, metric='olr', **params)

  return OLR_Dataset

def pyg_pe(dataset, **params):

  ''' TropD Precipitation minus Evaporation (PE) metric
     Var should contain one axis :class:`pyg.Lat`.  

     Parameters:
        pe(lat,): zonal-mean precipitation minus evaporation
   
        lat: equally spaced latitude column vector

        method (str): 
       
          'zero_crossing': the first latitude poleward of the subtropical minimum where P-E changes from negative to positive values. Only one method so far.
  
        lat_uncertainty (float, optional): The minimal distance allowed between the first and second zero crossings along lat
     
     Returns:
       PE_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)
  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> pe_data = -pyg.cosd(t2.lat*2)  #fake pe data
  >>> print(pyg_pe(pe_data))                                                                                                   
  <Var 'PE_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>> print(pyg_pe(pe_data)[:])                                                                                                
  [-45.  45.]
  '''     

  PE_Dataset = metrics_dataset(dataset, metric='pe', **params)

  return PE_Dataset

def pyg_psi(dataset,**params):

  ''' TropD Mass streamfunction (PSI) metric

  Latitude of the meridional mass streamfunction subtropical zero crossing
  
  Parameters:
  
    V(lat,lev): zonal-mean meridional wind
  
    lat: latitude vector

    lev: vertical level vector in hPa units
  
    method (str, optional):
  
      'Psi_500'(default): Zero crossing of the stream function (Psi) at the 500hPa level

      'Psi_500_10Perc': Crossing of 10# of the extremum value of Psi in each hemisphre at the 500hPa level

      'Psi_300_700': Zero crossing of Psi vertically averaged between the 300hPa and 700 hPa levels

      'Psi_500_Int': Zero crossing of the vertically-integrated Psi at the 500 hPa level

      'Psi_Int'    : Zero crossing of the column-averaged Psi
  
    lat_uncertainty (float, optional): The minimal distance allowed between the first and second zero crossings. For example, for lat_uncertainty = 10, the function will return a NaN value if a second zero crossings is found within 10 degrees of the most equatorward zero crossing.   

     Returns:
       PSI_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)
  
  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> V_data = -pyg.sind(t2.lat*6) * pyg.cos(t2.pres/1e3)  # fake meridional wind data
  >>> print(V_data)
  <Var '(-sind(lat)*cos(pres))'>:
    Shape:  (lat,pres)  (31,20)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
    Attributes:
      {}
    Type:  Mul_Var (dtype="float64")
  >>> print(pyg_psi(V_data)
  <Var '(-sind(lat)*cos(pres))'>:
    Shape:  (lat,pres)  (31,20)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
    Attributes:
      {}
    Type:  Mul_Var (dtype="float64")
   >>> print(pyg_psi(V_data)[:])                                                                                                
   [-30.  30.]
  '''

  PSI_Dataset = metrics_dataset(dataset, metric='psi', **params)

  return PSI_Dataset

def pyg_psl(dataset,**params):

  ''' TropD Sea-level pressure (PSL) metric
      Latitude of maximum of the subtropical sea-level pressure
      Var should contain one axis :class:`pyg.Lat`.
     
     Parameters
        ps(lat,): sea-level pressure
      
        lat: equally spaced latitude column vector

        method (str, optional): 'peak' (default) | 'max'
     
     Returns:
       PSL_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)

  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> psl_data = pyg.cosd(t2.lat*6)+1 #fake psl data
  >>> print(psl_data)                                                                                                     
  <Var '(cosd(lat)+1)'>:
    Shape:  (lat)  (31)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
    Attributes:
      {}
    Type:  Add_Var (dtype="float64")
  >>> print(pyg_psl(psl_data))                                                                                                
  <Var 'PSL_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>>  print(pyg_psl(psl_data)[:])                                                                                             
  [-53.99926851  53.99926851]
  '''

  PSL_Dataset = metrics_dataset(dataset, metric='psl', **params)

  return PSL_Dataset

def pyg_stj(dataset,**params):

  '''TropD Eddy Driven Jet (STJ) metric
       
     Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
     Var should contain axis :class:`pyg.Lat`. If :class:`pyg.Pres` is given, level closest to 850hPa is chosen 
     
     Parameters
        method (str, optional): 'peak' (default) |  'max' | 'fit'
       
        peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
        
        max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
        fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum
        
       n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
     
     Returns:
       STJ_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)

  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> U = t2.U(i_time=0).mean(pyg.Lon).squeeze()
  >>> print(U)
  <Var 'U'>:
  Shape:  (pres,lat)  (20,31)
  Axes:
    pres <Pres>    :  1000 hPa to 50 hPa (20 values)
    lat <Lat>      :  90 S to 90 N (31 values)
  Attributes:
    {}
  Type:  SqueezedVar (dtype="float64")
  >>> print(pyg_stj(U))
  <Var 'STJ_metrics'>:
  Shape:  (Metrics)  (2)
  Axes:
    Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
  Attributes:
    {}
  Type:  Replace_axes (dtype="float64")
  >>> print(pyg_stj(U)[:])                                                                                                    
  [-41.56747902  41.56747902]
  '''

  STJ_Dataset = metrics_dataset(dataset, metric='stj', **params)

  return STJ_Dataset

def pyg_tpb(dataset,**params):

  ''' TropD Tropopause break (TPB) metric
     Var should contain axes :class:`pyg.Lat`and :class:`pyg.Pres` 
     
     Parameters
        T(lat,lev): temperature (K)

        lat: latitude vector

        lev: pressure levels column vector in hPa

        method (str, optional): 
  
          'max_gradient' (default): The latitude of maximal poleward gradient of the tropopause height
  
          'cutoff': The most equatorward latitude where the tropopause crosses a prescribed cutoff value
  
          'max_potemp': The latitude of maximal difference between the potential temperature at the tropopause and at the surface
  
        Z(lat,lev) (optional): geopotential height (m)

        Cutoff (float, optional): geopotential height (m) cutoff that marks the location of the tropopause break
     
     Returns:
       TPB_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)

  '''

  TPB_Dataset = metrics_dataset(dataset, metric='tpb', **params)

  return TPB_Dataset

def pyg_uas(dataset,**params):

  ''' TropD near-surface zonal wind (UAS) metric
     Var should contain axis :class:`pyg.Lat. If :class:`pyg.Pres` is included, the nearest level to the surface is used.
     
     Parameters

        U(lat,lev) or U (lat,)-- Zonal mean zonal wind. Also takes surface wind 
        
        lat: latitude vector
        
        lev: vertical level vector in hPa units. lev=np.array([1]) for single-level input zonal wind U(lat,)

        method (str): 
          'zero_crossing': the first subtropical latitude where near-surface zonal wind changes from negative to positive

        lat_uncertainty (float, optional): the minimal distance allowed between the first and second zero crossings
     
     Returns:
       UAS_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)

  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> U_data = -pyg.cosd(2*t2.lat) * pyg.cos(t2.pres/1e3) #fake U data
  >>> print(U_data)                                                                                                       
  <Var '(-cosd(lat)*cos(pres))'>:
    Shape:  (lat,pres)  (31,20)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
    Attributes:
      {}
    Type:  Mul_Var (dtype="float64")
  >>> print(pyg_uas(U_data))                                                                                                  
  <Var 'UAS_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>> print(pyg_uas(U_data)[:])                                                                                               
  [-45.  45.]
  '''

  UAS_Dataset = metrics_dataset(dataset, metric='uas', **params)

  return UAS_Dataset

def pyg_gwl(dataset, **params):

  '''     
  '''     

  GWL_Dataset = metrics_dataset(dataset, metric='gwl', **params)

  return GWL_Dataset

def pyg_onesigma(dataset, **params):

  '''     
  '''     

  OneSigma_Dataset = metrics_dataset(dataset, metric='onesigma', **params)

  return OneSigma_Dataset

