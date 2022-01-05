import xarray as xr
import pytropd as pyt
import numpy as np

@xr.register_dataset_accessor("pyt_metrics")
class MetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.params = {}
        self.require_pres_axis = 0
        self.metric_name = None 
        self.xarray_dataset = None
        self.latitudes = None
        self.lat_name = None
        self.pres_name = None
        self.metric_property_name = dict(edj=['u', 'Zonal wind'],
                                         olr=['olr','Outgoing longwave radiation'],
                                         pe=['pe','Precipitation minus evaporation'],
                                         psi=['v','Meridional wind'],
                                         psl=['psl','Sea level pressure'],
                                         stj=['u','Zonal wind'],
                                         tpb=['T','Temperature'],
                                         uas=['uas','Surface wind'],
                                         )

    def metrics(self, data):
      """Return the PyTropD metrics this data."""
      
      # Compute the relevant metric
      metric_function = getattr(pyt, 'TropD_Metric_' + self.metric_name.upper()) 
      metric_lats = metric_function(data, lat=self.latitudes, **self.params)
      return np.squeeze(np.array(metric_lats))


    def extract_property_name(self, property_name): 
      
      if property_name[0] == 'lat':
        dataset_keys = list(self._obj.dims.keys())
        property_name_list = ['lat','latitude','lats','x','phi','degreesnorth']
        find_index = 1

      elif property_name[0] == 'pres':  
        dataset_keys = list(self._obj.dims.keys())
        property_name_list = ['pres','pressure','p','lev','levels','level']
        find_index = 1

      else:
        dataset_keys = list(self._obj.keys())

        #if we are only given one data array in the dataset, assume it is the right one,
        if len(dataset_keys) == 1:
          index = [0]
          find_index = 0

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
    
      if len(index)==0:
          print(self._obj)
          raise KeyError('%s not found in Dataset. Valid variable names are %s'%(property_name[1],property_name_list))

      if len(index)>1:
          print(self._obj)
          print('More than one possible key for %s found. Valid variable names are %s'%(property_name[1],property_name_list))
          raise KeyError

      print('Using %s in the dataset as the %s'%(dataset_keys[index[0]],property_name[1]))

      return dataset_keys[index[0]]
    
    def validate_data(self, property_name):   
      ## metrics that take 1D variable as input
      #if self.metric_name in ['olr','pe','psl']:
      #  max_axes = 1
      ##metrics that take 2D variable as input
      #elif self.metric_name in ['edj', 'psi','stj','tpb','uas']:
      #  max_axes = 2
      
      xarray_data = getattr(self._obj,self.extract_property_name(property_name=property_name))
      xarray_data = xarray_data.rename(self.metric_name)
      ndim = xarray_data.ndim
      self.xarray_dataset = xarray_data.to_dataset()
      ## check DataArray has axes (lat,) or (lat,pres)
      #try: 
      #  assert (ndim <= max_axes)
      #except AssertionError:
      #  print('Error: DataArray must only contains at most coordinates, Latitudes and Pressure')

      self.lat_name = self.extract_property_name(property_name=['lat','Latitude'])
      latitudes = getattr(self._obj, self.lat_name).values
      self.latitudes = latitudes
      #data_array = xarray_data.values

      if ndim >= 2 and self.require_pres_axis:
        #assume a pressure level is given and check for it.
        self.pres_name = self.extract_property_name(property_name=['pres','Pressure'])
        pressure = getattr(self._obj, self.pres_name).values
        self.params['lev'] = pressure

      #  #do we need to transpose values so that dimensions are (lat,pres)?
      #  if np.shape(data_array)[1] != len(pressure):
      #    data_array = data_array.transpose()
      #    assert np.shape(data_array)[1] == len(pressure)
      #self.data_values = data_array


    def edj(self,**params):
    
      '''TropD Eddy Driven Jet (EDJ) metric
           
         Latitude of maximum of the zonal wind at the level closest to the 850 hPa level
         Var should contain axis :class:`pyg.Lat`. If :class:`pyg.Pres` is given, level closest to 850hPa is chosen 
         
         Parameters
            method (str, optional): 'peak' (default) |  'max' | 'fit'
           
            peak (Default): Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=30)#
                                                                                                                                          #
            max: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level (smoothing parameter n=6)            #
            fit: Latitude of the maximum of the zonal wind at the level closest to the 850 hPa level using a quadratic polynomial fit of d#ata from gridpoints surrounding the gridpoint of the maximum
            
           n (int, optional): If n is not set (0), n=6 (default) is used in TropD_Calculate_MaxLat. Rank of moment used to calculate the position of max value. n = 1,2,4,6,8,...  
         
         Returns:
           EDJ_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)

      Examples
      --------
      '''
      #Validate data and extract data
      self.metric_name = 'edj'
      self.params = params
      self.require_pres_axis = 1
      self.validate_data(property_name=self.metric_property_name['edj'])
      edj_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,self.pres_name]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates

      return edj_lats.assign_coords(metrics=("metrics", np.array([0,1])))

    def olr(self, **params):
  
      """TropD Outgoing Longwave Radiation (OLR) metric
         
         Var should contain one axis :class:`pyg.Lat`.  
         Parameters:
         
           olr(lat,): zonal mean TOA olr (positive)
           
           lat: equally spaced latitude column vector
            
           method (str, optional):
  
             '250W'(Default): the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses 250W/m^2
             
      self.validate_data(property_name='uas')
             '20W': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses the tropical OLR max minus 20W/m^2
             
             'cutoff': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR crosses a specified cutoff value
             
             '10Perc': the first latitude poleward of the tropical OLR maximum in each hemisphere where OLR is 10# smaller than the tropical OLR maximum
             
             'max': the latitude of maximum of tropical olr in each hemisphere with the smoothing paramerer n=6 in TropD_Calculate_MaxLat
             
             'peak': the latitude of maximum of tropical olr in each hemisphere with the smoothing parameter n=30 in TropD_Calculate_MaxLat
           
           
           Cutoff (float, optional): Scalar. For the method 'cutoff', Cutoff specifies the OLR cutoff value. 
           
           n (int, optional): For the 'max' method, n is the smoothing parameter in TropD_Calculate_MaxLat
         
         Returns:
           OLR_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
      Examples
      --------
      """
  
      #Validate data and extract data
      self.metric_name = 'olr'
      self.params = params
      self.validate_data(property_name=self.metric_property_name['olr'])
      
      olr_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates
      metric_attrs = {'Description':'SH and NH latitudes'}
      return olr_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
  
    def pe(self, **params):
    
      ''' TropD Precipitation minus Evaporation (PE) metric
         Var should contain one axis :class:`pyg.Lat`.  
    
         Parameters:
            pe(lat,): zonal-mean precipitation minus evaporation
       
            lat: equally spaced latitude column vector
    
            method (str): 
           
              'zero_crossing': the first latitude poleward of the subtropical minimum where P-E changes from negative to positive values. Only one method so far.
      
            lat_uncertainty (float, optional): The minimal distance allowed between the first and second zero crossings along lat
         
         Returns:
           PE_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
      Examples
      --------
      '''     
      #Validate data and extract data
      self.metric_name = 'pe'
      self.params = params
      self.validate_data(property_name=self.metric_property_name['pe'])
      
      pe_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates
      metric_attrs = {'Description':'SH and NH latitudes'}

      return pe_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
    
    
    def psi(self,**params):
    
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
           PSI_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
    
      Examples
      --------
        '''
    
      #Validate data and extract data
      self.metric_name = 'psi'
      self.params = params
      self.require_pres_axis = 1
      self.validate_data(property_name=self.metric_property_name['psi'])
      psi_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,self.pres_name]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates

      metric_attrs = {'Description':'SH and NH latitudes'}
      return psi_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
      #Validate data and extract data
    
    
    def psl(self,**params):
    
      ''' TropD Sea-level pressure (PSL) metric
          Latitude of maximum of the subtropical sea-level pressure
          Var should contain one axis :class:`pyg.Lat`.
         
         Parameters
            ps(lat,): sea-level pressure
          
            lat: equally spaced latitude column vector
    
            method (str, optional): 'peak' (default) | 'max'
         
         Returns:
           PSL_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
    
      Examples
      --------
      '''
      #Validate data and extract data
      self.metric_name = 'psl'
      self.params = params
      self.validate_data(property_name=self.metric_property_name['psl'])
      
      psl_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates

      metric_attrs = {'Description':'SH and NH latitudes'}
      return psl_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
    
    
    def stj(self,**params):
    
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
           STJ_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
    
      Examples
      --------
      '''
      #Validate data and extract data
      self.metric_name = 'stj'
      self.params = params
      self.require_pres_axis = 1
      self.validate_data(property_name=self.metric_property_name['stj'])
      stj_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,self.pres_name]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates

      metric_attrs = {'Description':'SH and NH latitudes'}
      return stj_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
    
    
    def tpb(self,**params):
    
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
           TPB_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
    
      '''
      #Validate data and extract data
      self.metric_name = 'tpb'
      self.params = params
      self.require_pres_axis = 1
      self.validate_data(property_name=self.metric_property_name['tpb'])
      tbp_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,self.pres_name]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates
      metric_attrs = {'Description':'SH and NH latitudes'}

      return tbp_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
    
    
    def uas(self,**params):
    
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
           UAS_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
    
      Examples
      --------
      '''
      #Validate data and extract data
      self.metric_name = 'uas'
      self.params = params
      self.validate_data(property_name=self.metric_property_name['uas'])
      
      uas_lats = xr.apply_ufunc(self.metrics, 
                                self.xarray_dataset,  
                                input_core_dims=[[self.lat_name,]], 
                                output_core_dims=[["metrics"]], 
                                dask = 'allowed',
                                vectorize=True
                                ) 
      # define coordinates

      metric_attrs = {'Description':'SH and NH latitudes'}
      return uas_lats.assign_coords(metrics=("metrics", np.array([0,1]), metric_attrs))
    
    
    
