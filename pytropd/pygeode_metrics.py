import numpy as np
from scipy import interpolate
import pytropd as pyt

import pygeode as pyg
from pygeode.var import Var
from pygeode.dataset import Dataset, asdataset
from pygeode.axis import NamedAxis
from pygeode.tools import loopover, whichaxis 
from pygeode.view import View

from inspect import signature
import logging
from typing import Dict, List, Optional, Tuple

def metrics_dataset(
  dataset: pyg.Dataset, 
  metric: str, 
  **params
  ) -> pyg.Dataset:

  """Return a dataset of the metrics"""

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

  dataset, _ = extract_property(dataset, property_name=['lat','Latitude'], p_axis_status=p_axis_status)
  dataset, found_pres = extract_property(dataset, property_name=['pres','Pressure'], p_axis_status=p_axis_status)

  if found_pres and metric in ('edj','uas'):
    p_axis_status = 1


  #if dataset only contains one pyg.Var, assume
  #this is the correct variable for the metric and return it
  #otherwise, find check the name of the variable for the closest match
  # If none found, raise error.
  var, _ = extract_property(dataset, property_name=metric_property_name[metric], p_axis_status=p_axis_status)

  #nh hemisphere
  nh_data = chop_by_hemisphere(var,hem='nh')
  nh_var = metric_var(nh_data, output='lat', p_axis_status=p_axis_status, metric=metric, hem='nh', **params)
  
  #sh hemisphere
  sh_data = chop_by_hemisphere(var, hem='sh')
  sh_var = metric_var(sh_data, output='lat', p_axis_status=p_axis_status, metric=metric, hem='sh', **params)
 
  global_attrs = {
		  "long_name": metric.upper() + " metric latitude",
		  "units": "degrees",
		  }

  return pyg.Dataset(sh_var + nh_var, atts=global_attrs)


def metric_var(
  X: pyg.Var,
  output: str ='lat',  axis: Optional[int] = None,
  p_axis_status: Optional[int] = None, 
  metric: str = None, 
  hem: str = 'nh', 
  pbar: Optional[int] = None,
  **params
  ) ->  pyg.Var:

  '''Compute the metrics'''
  # Get the relevant metric function
  metric_function = getattr(pyt, 'TropD_Metric_' + metric.upper()) 

  # if method not provided, get default method of metric_function
  method_used: str = params.get(
	   "method", signature(metric_function).parameters["method"].default)
  
  ovars = ['lat','values']
  output = [o for o in output.split(',') if o in ovars]
  if len(output) < 1: raise ValueError('No valid outputs are requested from metric calculation. Possible outputs are %s.' % str(ovars))

  xn = X.name if X.name != '' else 'X' # Note: could write:  xn = X.name or 'X'

  #Re-order axes so that Lat and pres are at the end
  inaxes = list(X.axes)
  if X.hasaxis('Lat'):
    has_pres_axis = 0
    lat_axis_index = X.whichaxis('Lat')

    if X.hasaxis('Pres') and (p_axis_status==1):
      has_pres_axis = 1
    
      pres_axis_index = X.whichaxis('Pres')
      #out_order has lat, pres at the end
      out_order = [i for i in range(len(inaxes)) if i not in [pres_axis_index, lat_axis_index]]
      out_order.append(lat_axis_index)
      out_order.append(pres_axis_index)
    
    else:
      if p_axis_status==1:
        raise KeyError('<Pres> axis not found in', X)
      else:
        #out_order has lat at the end
        out_order = [i for i in range(len(inaxes)) if i not in [lat_axis_index,]]
        out_order.append(lat_axis_index)

  else: 
    raise KeyError('<Lat> axis not found in', X)

  #transpose axes in the order (Lat, Pres)
  X = X.transpose(*out_order)

  inaxes = list(X.axes)
  lataxis = X.whichaxis('Lat')
  lat_values = X.axes[lataxis][:]
  
  riaxes = [lataxis]
  if p_axis_status==1:
    presaxis = X.whichaxis('Pres')
    lev = X.axes[presaxis][:]
    params['lev'] = lev
    riaxes.append(presaxis)

  #construct outaxes 
  oaxes = [a for i, a in enumerate(X.axes) if i not in riaxes]
  if not oaxes:
    new_const_axis = NamedAxis(values=np.arange(1),name='value')
    X = X.extend(0, new_const_axis)
    oaxes.append(new_const_axis)
    inaxes = list(X.axes)

  # Construct new variable
  if hem == 'nh':
    name = metric.upper() + '_nh_latitude'
  else: 
    name = metric.upper() + '_sh_latitude'

  oview = View(oaxes) 

  if pbar is None:
    from pygeode.progress import PBar
    pbar = PBar(message='Computing ' + hem.upper())

  outview = View(oaxes)

  # Construct work arrays
  metric_lat= np.full(oview.shape, np.nan, 'd')
  metric_value = np.full(oview.shape, np.nan, 'd')

  
  # Accumulate data
  for outsl, (xdata,) in loopover([X], oview, inaxes, pbar=pbar):
    xdata = xdata.astype('d')
    
    if method_used =='fit':
      metric_lat[outsl], metric_value[outsl] = metric_function(xdata, lat_values, **params)
    else:
      metric_lat[outsl] = np.squeeze(np.array(metric_function(xdata, lat_values, **params)))

  var_list_out = []

  if 'lat' in output:
    lat_attrs = {"long_name": metric.upper() + " metric latitude",
          	 "unit": "degrees",
	  	 "method_used:": method_used,
	         }
    metric_lat = Var(oaxes, values=metric_lat, name=hem + '_metric_lat', atts=lat_attrs)
    var_list_out.append(metric_lat)

  pbar.update(100)
  return var_list_out



def pressure_axis_status(metric: str) -> int:
  '''Decide if computation of metric requires a pressure axis'''
  # metrics that take 1D variable as input
  if metric in ['edj','olr','pe','psl','uas']:
    return 0

  #metrics that takes a 2D variable as input
  elif metric in ['psi','stj','tpb']:
    return 1
  
  #metrics that can take a 2D variable as input but do not collapse pressure axis
  elif metric in ['gwl', 'onesigma']:
    return 2

def extract_property(
  dataset: pyg.Dataset, 
  property_name: List[str], 
  p_axis_status: int
  ) -> Tuple[pyg.Dataset,int]: 
  '''
  Search the dataset for the name of the variable required by the metric

  Parameters
  ----------
  property_name : List[str]
	  pair of variable names, one a short key and other long description

  Returns
  -------
	  pyg.Dataset with sanitised variable names for the metric
	  int: 1 if property has been found, 0 otherwise
  '''

  find_index = 0

  if property_name[0] == 'lat':
    #check if pyg.Lat axis is present, else look for a match
    if dataset.hasaxis(pyg.Lat):
      print('Found Latitude axis in the dataset')
      return dataset, 1
    
    else:
      dataset_keys = list(dataset.axisdict.keys())
      property_name_list = ['lat','latitude','lats','x','phi','degreesnorth']
      find_index = 1

  elif property_name[0] == 'pres':	
    #check if pyg.Pres axis is present, else look for a match
    if dataset.hasaxis(pyg.Pres):
      print('Found Pressure axis in the dataset')
      return dataset, 1
    
    else:
      #try to see if there is a pressure axis as a pyg.NamedAxis
      dataset_keys = list(dataset.axisdict.keys())
      property_name_list = ['pres','pressure','p','lev','plev','levels','level']
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
      elif property_name[0] == 'uas':
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
      return dataset.replace_axes({dataset_keys[index[0]]: pyg.Pres(pres_axis[:])}), 1
    
    else:
      return dataset, 0

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
    return dataset.replace_axes({dataset_keys[index[0]]: pyg.Lat(lat_axis[:])}), 1


  else:
    print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
    return getattr(dataset, dataset_keys[index[0]]),1
  
  
def chop_by_hemisphere(
    dataset: pyg.Dataset,
    hem: str = 'nh' 
    ) -> pyg.Dataset:

  '''Chop up dataset into NH and SH'''
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

def pyg_edj(dataset: pyg.Dataset,**params) -> pyg.Dataset:

  '''TropD Eddy Driven Jet (EDJ) metric

	 Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
	 Var should contain axis :class:`pyg.Lat`. If :class:`pyg.Pres` is given, level closest to 850hPa is chosen 

	 Parameters
		method (str, optional): 'peak' (default) |	'max' | 'fit'

		peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)

		max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)
		fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of data from gridpoints surrounding the gridpoint of the maximum

	   n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  

	 Returns:
	   EDJ_metrics: :class:Var` with axis :class:`ǸamedAxis` Metric (SH latitudes, NH latitudes)

  Examples
  --------
  >>> import pytropd as pyt 
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> U = t2.U(i_time=0).mean(pyg.Lon).squeeze()
  >>> print(U)
  <Var 'U'>:
	Shape:	(pres,lat)	(20,31)
  Axes:
	pres <Pres>    :  1000 hPa to 50 hPa (20 values)
	lat <Lat>	   :  90 S to 90 N (31 values)
  Attributes:
	{}
  Type:  SqueezedVar (dtype="float64")
  >>> edj_metric_dataset = pyt.pyg_edj(U)  #Calculate EDJ metric	 
  edj_metric_dataset = pyt.pyg_edj(U)
  Found Latitude axis in the dataset
  Found Pressure axis in the dataset
  Using U in the dataset as the Zonal wind
  >>> print(edj_metric_dataset)
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : EDJ metric latitude
	units		   : degrees
  >>> print(edj_metric_dataset.nh_metric_lat[:])
  [45.]
  '''

  EDJ_Dataset = metrics_dataset(dataset, metric='edj', **params)

  return EDJ_Dataset


def pyg_olr(dataset: pyg.Dataset, **params) -> pyg.Dataset:

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
	Shape:	(lat)  (31)
	Axes:
	  lat <Lat>		 :	90 S to 90 N (31 values)
	Attributes:
	  {}
	Type:  Add_Var (dtype="float64")
  >>> olr_metric_dataset = pyt.pyg_olr(olr_data) #Calculate OLR metric	   
  Found Latitude axis in the dataset
  Using ((cosd(lat)*150)+150) in the dataset as the Outgoing longwave radiation
  >>> print(olr_metric_dataset)																					   
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : OLR metric latitude
	units		   : degrees
  >>> print(olr_metric_dataset.nh_metric_lat[:])																   
  [24.0874096]
  """

  OLR_Dataset = metrics_dataset(dataset, metric='olr', **params)

  return OLR_Dataset

def pyg_pe(dataset: pyg.Dataset, **params) -> pyg.Dataset:

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
	Shape:	(Metrics)  (2)
	Axes:
	  Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
	Attributes:
	  {}
	Type:  Replace_axes (dtype="float64")
  >>> pe_metric_dataset = pyt.pyg_pe(pe_data) #Calculate PE metric											 
  Found Latitude axis in the dataset
  Using -cosd(lat) in the dataset as the Precipitation minus evaporation
  >>>print(pe_metric_dataset)																					 
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : PE metric latitude
	units		   : degrees

  >>> print(pe_metric_dataset.nh_metric_lat[:])																	  
  [45.]
  '''	  

  PE_Dataset = metrics_dataset(dataset, metric='pe', **params)

  return PE_Dataset

def pyg_psi(dataset: pyg.Dataset,**params) -> pyg.Dataset:

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
	Shape:	(lat,pres)	(31,20)
	Axes:
	  lat <Lat>		 :	90 S to 90 N (31 values)
	  pres <Pres>	 :	1000 hPa to 50 hPa (20 values)
	Attributes:
	  {}
	Type:  Mul_Var (dtype="float64")
  >>> psi_metric_dataset = pyt.pyg_psi(V_data)	 #Calculate PSI metric		  
  Found Latitude axis in the dataset
  Found Pressure axis in the dataset
  Using (-sind(lat)*cos(pres)) in the dataset as the Meridional wind
  >>> print(psi_metric_dataset)																					  
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : PSI metric latitude
	units		   : degrees
  >>> print(psi_metric_dataset.nh_metric_lat[:])		 
   [-30.]
  '''

  PSI_Dataset = metrics_dataset(dataset, metric='psi', **params)

  return PSI_Dataset

def pyg_psl(dataset: pyg.Dataset,**params) -> pyg.Dataset:

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
	Shape:	(lat)  (31)
	Axes:
	  lat <Lat>		 :	90 S to 90 N (31 values)
	Attributes:
	  {}
	Type:  Add_Var (dtype="float64")
  >>> psl_metric_dataset = pyt.pyg_psl(psl_data)  #Calculate PSL metric											  
  Found Latitude uaxis in the dataset
  Using (cosd(lat)+1) in the dataset as the Sea level pressure
  >>> print(psl_metric_dataset)																					  
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : PSL metric latitude
	units		   : degrees
  >>> print(psl_metric_dataset.nh_metric_lat[:])																  
  [53.99926851]
  '''

  PSL_Dataset = metrics_dataset(dataset, metric='psl', **params)

  return PSL_Dataset

def pyg_stj(dataset: pyg.Dataset,**params) -> pyg.Dataset:

  '''TropD Eddy Driven Jet (STJ) metric
	   
	 Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
	 Var should contain axis :class:`pyg.Lat`. If :class:`pyg.Pres` is given, level closest to 850hPa is chosen 
	 
	 Parameters
		method (str, optional): 'peak' (default) |	'max' | 'fit'
	   
		peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)
		u
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
	lat <Lat>	   :  90 S to 90 N (31 values)
  Attributes:
	{}
  Type:  SqueezedVar (dtype="float64")
  >>> stj_metric_dataset = pyt.pyg_stj(U) #Calculate STJ metric
  Found Latitude axis in the dataset
  Found Pressure axis in the dataset
  Using U in the dataset as uthe Zonal wind
  >>> print(stj_metric_dataset)																					  
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : STJ metric latitude
	units		   : degrees
  >>> print(stj_metric_dataset.nh_metric_lat[:])																  
  [41.56747902]
  '''

  STJ_Dataset = metrics_dataset(dataset, metric='stj', **params)

  return STJ_Dataset

def pyg_tpb(dataset: pyg.Dataset,**params) -> pyg.Dataset:

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

  Examples
  --------
  >>> import pygeode as pyg
  >>> from pygeode.tutorial import t2 
  >>> T = t2.Temp(i_time=0).mean(pyg.Lon).squeeze() 
  >>> tpb_metric_dataset = pyt.pyg_tpb(T)  #Calculate TBP metric
  Found Latitude axis in the dataset
  Found Pressure axis in the dataset
  Using Temp in the dataset as the Temperature
  >>> print(tpb_metric_dataset)																					  
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : TPB metric latitude
	units		   : degrees
  '''

  TPB_Dataset = metrics_dataset(dataset, metric='tpb', **params)

  return TPB_Dataset

def pyg_uas(dataset: pyg.Dataset,**params) -> pyg.Dataset:

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
	Shape:	(lat,pres)	(31,20)
	Axes:
	  lat <Lat>		 :	90 S to 90 N (31 values)
	  pres <Pres>	 :	1000 hPa to 50 hPa (20 values)
	Attributes:
	  {}
	Type:  Mul_Var (dtype="float64")
  >>> uas_metric_dataset = pyt.pyg_uas(U_data) #Calculate UAS metric
  Found Latitude axis in the dataset
  Found Pressure axis in the dataset
  Using (-cosd(lat)*cos(pres)) in the dataset as the Surface wind
  >>> print(uas_metric_dataset)																					  
  <Dataset>:
  Vars:
	sh_metric_lat (value)  (1)
	nh_metric_lat (value)  (1)
  Axes:
	value <NamedAxis 'value'>:	0 
  Global Attributes:
	long_name	   : UAS metric latitude
	units		   : degrees
  >>> print(uas_metric_dataset.nh_metric_lat[:])																  
  [45.]
  '''

  UAS_Dataset = metrics_dataset(dataset, metric='uas', **params)

  return UAS_Dataset

