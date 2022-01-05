from pygeode.var import Var
import numpy as np
import pytropd as pyt
from pygeode.axis import NamedAxis
import pygeode as pyg

class MetricVar (Var):
  ''''''

  def __init__ (self, data, metric=None, need_pres_axis=0, **params):
    from pygeode.var import Var
    '''__init__()'''
    self.metric_property_name = dict(edj=['u', 'Zonal wind'],
                                olr=['olr','Outgoing longwave radiation'],
                                pe=['pe','Precipitation minus evaporation'],
                                psi=['v','Meridional wind'],
                                psl=['psl','Sea level pressure'],
                                stj=['u','Zonal wind'],
                                tpb=['T','Temperature'],
                                uas=['uas','Surface wind'],
                                )

    self.metric = metric

    self.needs_paxis = self.pressure_axis_needed()
    data = self.extract_property_name(data, property_name=['lat','Latitude'])
    data = self.extract_property_name(data, property_name=['pres','Pressure'])
    var = self.find_relevant_variable(data)

    self.var = var.squeeze()
    self.params = params
    
    axes = list(self.var.axes)
    out_order = [type(i) for i in axes]

    if var.hasaxis('Lat'):
      self.has_pres_axis = 0
      
      #move lat axis to the end
      out_order.append(out_order.pop(var.whichaxis('Lat')))

      if var.hasaxis('Pres'):
        #move pres axis to the end
        out_order.append(out_order.pop(var.whichaxis('Pres')))
        self.has_pres_axis = 1

      else:
        if self.needs_paxis:
          print(var)
          raise KeyError('<Pres> axis not found in', var)
   
    else: 
      print(var)
      raise KeyError('<Lat> axis not found in', var)

    
    #transpose axes in the order (Lat, Pres)
    self.var = self.var.transpose(*out_order)
    
    lataxis = self.var.whichaxis('Lat')
    self.lat_values = self.var.axes[lataxis][:]

    self.riaxes = [lataxis]
    if self.has_pres_axis:
      presaxis = self.var.whichaxis('Pres')
      self.lev = self.var.axes[presaxis][:]
      self.params['lev'] = self.lev
      self.riaxes.append(presaxis)
    
    self.raxes = [a for i, a in enumerate(self.var.axes) if i in self.riaxes]
    self.oaxes = [a for i, a in enumerate(self.var.axes) if i not in self.riaxes]

    self.metric_axis = NamedAxis(values=np.arange(2),name='Metrics')
    self.outaxes = self.oaxes.copy()
    self.outaxes.append(self.metric_axis)

    # Construct new variable
    name = metric.upper() + '_metrics'
    Var.__init__(self, self.outaxes, var.dtype, name=name, atts=var.atts, plotatts=var.plotatts)

    
  def metric_function(self, data):
    # Compute the relevant metric:ta
    metricfunction = getattr(pyt, 'TropD_Metric_' + self.metric.upper()) 
    metric_lats = metricfunction(data, lat=self.lat_values, **self.params)
    return np.squeeze(np.array(metric_lats))


  def getview (self, view, pbar):
  
    from pygeode.tools import whichaxis, loopover
    from pygeode.view import View
    self.oview = View(self.oaxes) 
    self.iview = View(self.raxes) 

    if pbar is None:
      from pygeode.progress import PBar
      pbar = PBar()

    outview = View(self.outaxes)

    #Metric axis is the last in the view
    #Remove from the view for loopover
    loopover_view = view.remove(len(view.shape)-1)
    # Construct work arrays
    x = np.zeros(loopover_view.shape + (2,), 'd')

    x[()] = np.nan

    # Vectorize the metric function depending on inputs needed
    if self.has_pres_axis:
      signature = '(i,j)->(k)'
    else:
      signature = '(i)->(k)'

    metric_function_v = np.vectorize(self.metric_function, signature=signature)
  
    
    # Accumulate data
    for outsl, (xdata,) in loopover([self.var], loopover_view, preserve=self.riaxes, pbar=pbar):
      outsl = outsl + (slice(0,2,1),)
      xdata = xdata.astype('d')
      x[outsl] = metric_function_v(xdata)

    maxis = self.whichaxis(self.metric_axis)
    metric_values_requested = view.subaxis(maxis)[:]
    return x[...,metric_values_requested]

  def pressure_axis_needed(self):
    # metrics that take 1D variable as input
    if self.metric in ['edj','olr','pe','psl','uas']:
      return 0

    #metrics that take 2D variable as input
    elif self.metric in ['psi','stj','tpb']:
      return 1
  
  def extract_property_name(self, data, property_name): 
  
    find_index = 0
  
    if property_name[0] == 'lat':
      #check if pyg.Lat axis is present, else look for a match
      if data.hasaxis(pyg.Lat):
        print('Found Latitude axis in the dataset')
        return data
  
      else:
        dataset_keys = list(data.axisdict.keys())
        property_name_list = ['lat','latitude','lats','x','phi','degreesnorth']
        find_index = 1
  
    elif property_name[0] == 'pres':  
      #check if pyg.Pres axis is present, else look for a match
      if data.hasaxis(pyg.Pres):
        print('Found Pressure axis in the dataset')
        return data
  
      else:
        #try to see if there is a pressure axis as a pyg.NamedAxis
        dataset_keys = list(data.axisdict.keys())
        property_name_list = ['pres','pressure','p','lev','levels','level']
        find_index = 1
  
    else:
      dataset_keys = list(data.vardict.keys())
  
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
      if self.needs_paxis and len(index)==0: 
        print(data)
        raise KeyError('%s not found in Dataset. Valid variable names are %s'%(property_name[1],property_name_list))
  
      elif len(index)==1:
        #if we find a pres axis convert it to a pyg.Pres axis and return data
        print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
        pres_axis = getattr(data, dataset_keys[index[0]])
        print('Replacing pyg.NamedAxis %s with a pyg.Pres axis in the dataset'%(dataset_keys[index[0]]))
        return data.replace_axes({dataset_keys[index[0]]: pyg.Pres(pres_axis[:])})

      else:
        return data
  
    if len(index)==0: 
      print(data)
      raise KeyError('%s not found in Dataset. Valid variable names are %s'%(property_name[1],property_name_list))
  
  
    if len(index)>1:
      print(data)
      print('More than one possible key for %s found. Valid variable names are %s'%(property_name[1],property_name_list))
      raise KeyError
  
    if property_name[0] == 'lat' and find_index:
      print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
      lat_axis = getattr(data, dataset_keys[index[0]])
      print('Replacing pyg.NamedAxis %s with a pyg.Lat axis in the dataset'%(dataset_keys[index[0]]))
      return data.replace_axes({dataset_keys[index[0]]: pyg.Lat(lat_axis[:])})
  
  
    else:
      print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))
      return getattr(data, dataset_keys[index[0]])
  
  
  def find_relevant_variable(self, data):
  
    #if pyg.Var given, return this
    if isinstance(data, pyg.Var): return data
  
    #if data is a pyg.Dataset and only contains one pyg.Var, assume
    #this is the correct variable for the metric and return it
    #otherwise, find check the name of the variable for the closest match
    # If none found, raise error.
    if isinstance(data, pyg.Dataset):
      return self.extract_property_name(data, property_name=self.metric_property_name[self.metric])



def edj(data,**params):

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
  >>> print(edj(U))
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
  EDJ_Var = MetricVar(data, metric='edj', **params)

  return EDJ_Var


def olr(var, **params):

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


  OLR_Var = MetricVar(var, metric='olr', **params)

  return OLR_Var

def pe(var, **params):

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
  >>> print(pe(pe_data))                                                                                                   
  <Var 'PE_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>> print(pe(pe_data)[:])                                                                                                
  [-45.  45.]
  '''     

  PE_Var = MetricVar(var, metric='pe', **params)

  return PE_Var

def psi(var,**params):

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
  >>> print(psi(V_data)
  <Var '(-sind(lat)*cos(pres))'>:
    Shape:  (lat,pres)  (31,20)
    Axes:
      lat <Lat>      :  90 S to 90 N (31 values)
      pres <Pres>    :  1000 hPa to 50 hPa (20 values)
    Attributes:
      {}
    Type:  Mul_Var (dtype="float64")
   >>> print(psi(V_data)[:])                                                                                                
   [-30.  30.]
  '''

  PSI_Var = MetricVar(var, metric='psi', **params)

  return PSI_Var

def psl(var,**params):

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
  >>> print(psl(psl_data))                                                                                                
  <Var 'PSL_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>>  print(psl(psl_data)[:])                                                                                             
  [-53.99926851  53.99926851]
  '''

  PSL_Var = MetricVar(var, metric='psl', **params)

  return PSL_Var

def stj(var,**params):

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
  >>> print(stj(U))
  <Var 'STJ_metrics'>:
  Shape:  (Metrics)  (2)
  Axes:
    Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
  Attributes:
    {}
  Type:  Replace_axes (dtype="float64")
  >>> print(stj(U)[:])                                                                                                    
  [-41.56747902  41.56747902]
  '''

  STJ_Var = MetricVar(var, metric='stj', **params)

  return STJ_Var

def tpb(var,**params):

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

  TPB_Var = MetricVar(var, metric='tpb', **params)

  return TPB_Var

def uas(var,**params):

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
  >>> print(uas(U_data))                                                                                                  
  <Var 'UAS_metrics'>:
    Shape:  (Metrics)  (2)
    Axes:
      Metrics <NamedAxis 'Metrics'>:  0  to 1  (2 values)
    Attributes:
      {}
    Type:  Replace_axes (dtype="float64")
  >>> print(uas(U_data)[:])                                                                                               
  [-45.  45.]
  '''

  UAS_Var = MetricVar(var, metric='uas', **params)

  return UAS_Var


