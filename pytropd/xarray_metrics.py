from typing import Callable, Dict, List, Sequence, Any
import xarray as xr
import pytropd as pyt
import numpy as np
import logging


@xr.register_dataset_accessor("pyt_metrics")
class MetricAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self.params: Dict[str, Any] = {}
        self.require_pres_axis = False
        self.metric_name = ""
        self.xarray_dataset = xr.Dataset()
        self.latitudes = np.array([])
        self.lat_name = ""
        self.pres_name = ""
        self.metric_property_name = dict(
            edj=["u", "Zonal wind"],
            olr=["olr", "Outgoing longwave radiation"],
            pe=["pe", "Precipitation minus evaporation"],
            psi=["v", "Meridional wind"],
            psl=["psl", "Sea level pressure"],
            stj=["u", "Zonal wind"],
            tpb=["T", "Temperature"],
            uas=["uas", "Surface wind"],
        )

    def metrics(self, data: np.ndarray) -> Callable:
        """Return the self.pytropD metrics this data."""

        # Compute the relevant metric
        metric_function = getattr(pyt, "TropD_Metric_" + self.metric_name.upper())
        return metric_function(data, lat=self.latitudes, **self.params)

    def extract_property_name(self, property_name: List[str]) -> str:
        """
        search the dataset for the name of the variable required by the metric

        Parameters
        ----------
        property_name : List[str]
            pair of variable names, one a short key and other long description

        Returns
        -------
        str
            the variable name matching the required variable for the metric
        """

        def join_strlist(mylist: Sequence[str], joiner: str = "and") -> str:
            """
            helper func for nicely formatted lists

            Parameters
            ----------
            mylist : Sequence[str]
                sequence of strings to join
            joiner : str, optional
                word for joining last item, by default "and"

            Returns
            -------
            str
                joined list as single string
            """

            if len(mylist) < 2:
                try:
                    return mylist[0]
                except IndexError:
                    return ""
            else:
                return f"{', '.join(mylist[:-1])} {joiner} {mylist[-1]}"

        lookup_property_names = {
            "lat": ["lat", "latitude", "lats", "x", "phi", "degreesnorth"],
            "pres": ["pres", "pressure", "p", "lev", "levels", "level"],
            "u": ["zonalwind", "uwind", "u", "xwind"],
            "uas": ["surfacewind", "uas", "us", "surfu", "usurf"],
            "v": ["meridionalwind", "vwind", "v", "ywind"],
            "T": ["t", "temp", "temperature"],
            "psl": ["sealevelpressure", "slp", "psl", "ps", "sp"],
            "olr": ["olr", "outgoinglongwaveradiation", "toaolr", "olrtoa"],
            "pe": ["pe", "precipitationminusevarporation", "pminuse"],
        }
        short_property_name, long_property_name = property_name

        if short_property_name in ["lat", "pres"]:
            dataset_keys = [str(d) for d in self._obj.dims]
        else:
            dataset_keys = [str(d) for d in self._obj.data_vars]

        # if we are only given one data array in the dataset, assume it is the right one,
        if len(dataset_keys) == 1:
            logging.debug(
                f"Using {dataset_keys[0]} in the dataset as the " + long_property_name
            )
            return dataset_keys[0]

        # otherwise look for a matching variable name
        # Remove whitespace, underscores and make lowercase
        stripped_keys = [
            key.strip().lower().replace("_", "").replace("-", "")
            for key in dataset_keys
        ]
        # find variable
        all_property_names = lookup_property_names[short_property_name]
        matched_keys = set(stripped_keys) & set(all_property_names)
        if len(matched_keys) > 1:
            raise KeyError(
                f"More than one possible key for {long_property_name} found. Detected "
                f"variables are {join_strlist(dataset_keys)}. Expected variable names"
                f" are {join_strlist(all_property_names, joiner='or')}."
            )

        # extract relevant index
        indexes = [stripped_keys.index(key) for key in matched_keys]

        try:
            property_key = dataset_keys[indexes[0]]
        except IndexError:
            raise KeyError(
                f"{long_property_name} not found in Dataset. Detected variables are "
                f"{join_strlist(dataset_keys)}. Expected variable names are "
                f"{join_strlist(all_property_names, joiner='or')}."
            )

        logging.debug(
            f"Using {property_key} in the dataset as the {long_property_name}"
        )

        return property_key

    def validate_data(self):

        property_name = self.metric_property_name[self.metric_name]

        xarray_data: xr.DataArray = self._obj[
            self.extract_property_name(property_name=property_name)
        ]
        xarray_data = xarray_data.rename(self.metric_name)
        self.xarray_dataset = xarray_data.to_dataset()

        self.lat_name = self.extract_property_name(property_name=["lat", "Latitude"])
        latitudes: np.ndarray = xarray_data[self.lat_name].values
        self.latitudes = latitudes

        try:
            self.pres_name = self.extract_property_name(
                property_name=["pres", "Pressure"]
            )
        except KeyError as e:
            if self.require_pres_axis:
                raise e

        # check for pressure if required or present for optional ones
        if self.require_pres_axis or (
            self.metric_name in ["edj", "uas"] and self.pres_name
        ):
            pressure: np.ndarray = xarray_data[self.pres_name].values
            self.params["lev"] = pressure
        # otherwise don't use it if it is present
        else:
            self.pres_name = ""
        return

    def xr_edj(self, **params) -> xr.Dataset:

        """TropD Eddy Driven Jet (EDJ) metric

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
        """
        # Validate data and extract data
        self.metric_name = "edj"
        self.params = params
        self.validate_data()
        edj_latSH, edj_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[
                [self.lat_name, self.pres_name] if self.pres_name else [self.lat_name]
            ],
            output_core_dims=[[], []],
            dask="allowed",
        )
        edj_lats = xr.concat([edj_latSH, edj_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return edj_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_olr(self, **params) -> xr.Dataset:

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
             OLR_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)
        Examples
        --------
        """

        # Validate data and extract data
        self.metric_name = "olr"
        self.params = params
        self.validate_data()

        olr_latSH, olr_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[[self.lat_name]],
            output_core_dims=[[], []],
            dask="allowed",
        )
        olr_lats = xr.concat([olr_latSH, olr_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return olr_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_pe(self, **params) -> xr.Dataset:

        """TropD Precipitation minus Evaporation (PE) metric
             Var should contain one axis :class:`pyg.Lat`.

        import xarray as xr
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
        """
        # Validate data and extract data
        self.metric_name = "pe"
        self.params = params
        self.validate_data()

        pe_latSH, pe_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[[self.lat_name]],
            output_core_dims=[[], []],
            dask="allowed",
        )
        pe_lats = xr.concat([pe_latSH, pe_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return pe_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_psi(self, **params) -> xr.Dataset:

        """TropD Mass streamfunction (PSI) metric

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
        """

        # Validate data and extract data
        self.metric_name = "psi"
        self.params = params
        self.require_pres_axis = True
        self.validate_data()
        psi_latSH, psi_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[[self.lat_name, self.pres_name]],
            output_core_dims=[[], []],
            dask="allowed",
        )
        psi_lats = xr.concat([psi_latSH, psi_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return psi_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_psl(self, **params) -> xr.Dataset:

        """TropD Sea-level pressure (PSL) metric
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
        """
        # Validate data and extract data
        self.metric_name = "psl"
        self.params = params
        self.validate_data()

        psl_latSH, psl_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[[self.lat_name]],
            output_core_dims=[[], []],
            dask="allowed",
        )
        psl_lats = xr.concat([psl_latSH, psl_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return psl_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_stj(self, **params) -> xr.Dataset:

        """TropD Eddy Driven Jet (STJ) metric

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
        """
        # Validate data and extract data
        self.metric_name = "stj"
        self.params = params
        self.require_pres_axis = True
        self.validate_data()
        stj_latSH, stj_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[[self.lat_name, self.pres_name]],
            output_core_dims=[[], []],
            dask="allowed",
        )
        stj_lats = xr.concat([stj_latSH, stj_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return stj_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_tpb(self, **params) -> xr.Dataset:

        """TropD Tropopause break (TPB) metric
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

        """
        # Validate data and extract data
        self.metric_name = "tpb"
        self.params = params
        self.require_pres_axis = True
        self.validate_data()
        tpb_latSH, tpb_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[[self.lat_name, self.pres_name]],
            output_core_dims=[[], []],
            dask="allowed",
        )
        tpb_lats = xr.concat([tpb_latSH, tpb_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return tpb_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )

    def xr_uas(self, **params) -> xr.Dataset:

        """TropD near-surface zonal wind (UAS) metric
           Var should contain axis :class:`pyg.Lat. If :class:`pyg.Pres` is included, the nearest level to the surface is used.

           Parameters

              U(lat,lev) or U (lat,)-- Zonal mean zonal wind. Also takes surface wind

              lat: latitude vector

              lev: vertical level vector in hPa units. lev=np.array([1]) for single-level iself.nput zonal wind U(lat,)

              method (str):
                'zero_crossing': the first subtropical latitude where near-surface zonal wind changes from negative to positive

              lat_uncertainty (float, optional): the minimal distance allowed between the first and second zero crossings

           Returns:
             UAS_metrics: :class:'xarray.Dataset` with dimensions :property:`xarray.Dataset.Coords` metric_dim (SH latitudes, NH latitudes)

        Examples
        --------
        """
        # Validate data and extract data
        self.metric_name = "uas"
        self.params = params
        self.validate_data()

        uas_latSH, uas_latNH = xr.apply_ufunc(
            self.metrics,
            self.xarray_dataset,
            input_core_dims=[
                [self.lat_name, self.pres_name] if self.pres_name else [self.lat_name]
            ],
            output_core_dims=[[], []],
            dask="allowed",
        )
        uas_lats = xr.concat([uas_latSH, uas_latNH], dim="metrics")
        # define coordinates
        metric_attrs = {"Description": "SH and NH latitudes"}
        return uas_lats.assign_coords(
            metrics=("metrics", np.array([0, 1]), metric_attrs)
        )
