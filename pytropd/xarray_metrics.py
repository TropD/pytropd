from typing import Dict, List, Sequence, Any, Tuple, Union
import xarray as xr
import pytropd as pyt
import numpy as np
import logging
from inspect import signature


@xr.register_dataset_accessor("pyt_metrics")
class MetricAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self.params: Dict[str, Any] = {}
        self.require_pres_axis = False
        self.metric_name = ""
        self.xarray_data = xr.DataArray()
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

    def compute_metrics(self) -> xr.DataArray:
        """Return the requested pytropd metric for this data"""

        self.validate_data()

        # Compute the relevant metric
        metric_function = getattr(pyt, "TropD_Metric_" + self.metric_name.upper())
        # if method not provided, get default method of metric_function
        method_used: str = self.params.get(
            "method", signature(metric_function).parameters["method"].default
        )
        func_returns_vals = (self.metric_name in ["edj", "stj"]) and (
            method_used == "fit"
        )
        input_core_dims = [[self.lat_name]]
        # validate_data ensures self.pres_name is only defined if the metric
        # should accept it as an arg or kwarg
        if self.pres_name:
            input_core_dims[0].append(self.pres_name)
        has_nhem = (self.latitudes > 20.0).any()
        has_shem = (self.latitudes < -20.0).any()
        nhems = int(has_nhem) + int(has_shem)
        if nhems == 0:
            raise ValueError("not enough latitudes were provided")
        nreturns = 2 * nhems if func_returns_vals else nhems

        # we need to ensure Z is aligned properly with T when doing TropD_Metric_TPB and
        # method="cutoff"
        metric_output: Union[xr.DataArray, Tuple[xr.DataArray, ...]]
        if (self.metric_name == "tpb") and ("Z" in self.params):
            Z = self.params.pop("Z")
            try:
                metric_output = xr.apply_ufunc(
                    lambda data, Z, **kwargs: metric_function(data, Z=Z, **kwargs)[
                        slice(None) if nreturns > 1 else 0
                    ],
                    self.xarray_data,
                    Z,
                    kwargs=self.params,
                    input_core_dims=2 * input_core_dims,
                    output_core_dims=nreturns * [[]],
                    # we might want to warn users when using dask as some pytropd functions
                    # still iterate over arrays under the hood, which is very slow with dask
                    dask="allowed",
                )
            except Exception as e:
                print(e)
                breakpoint()
        else:
            metric_output = xr.apply_ufunc(
                metric_function
                if nreturns != 1
                else lambda data, **kwargs: metric_function(data, **kwargs)[0],
                self.xarray_data,
                kwargs=self.params,
                input_core_dims=input_core_dims,
                output_core_dims=nreturns * [[]],
                # we might want to warn users when using dask as some pytropd functions
                # still iterate over arrays under the hood, which is very slow with dask
                dask="allowed",
            )

        if isinstance(metric_output, tuple):
            metric_lats = xr.concat(metric_output[:nhems], dim="hemsph")
        else:
            metric_lats = metric_output.expand_dims("hemsph")
        # define coordinates, ensuring all added coords follow CF conventions
        hemsph_attrs = {"long_name": "hemisphere", "units": ""}

        # now let's provide a brief description of the metric from the docs
        # all methods are in bulleted lists, so split on the reST bullet
        # and get rid of extra stuff at beginning and end
        parsed_doc = str(metric_function.__doc__).split("* ")[1:]
        parsed_doc[-1] = parsed_doc[-1].split("\n\n")[0]
        # now we need to get the methods description by cleaning the method name
        method_desc = {
            m.split(": ")[0].strip('"'): m.split(": ")[1] for m in parsed_doc
        }.get(method_used, "")
        method_attrs = {
            "long_name": "method used",
            "units": "",
            "description": method_desc,
        }

        # finally make the output variable CF compliant
        metric_lats.attrs = {
            "long_name": self.metric_name.upper() + " metric latitude",
            "unit": self._obj[self.lat_name].attrs.get("unit", "degrees"),
        }
        hemsph = []
        if has_shem:
            hemsph.append("SH")
        if has_nhem:
            hemsph.append("NH")
        metric_lats = metric_lats.expand_dims("method").assign_coords(
            hemsph=("hemsph", hemsph, hemsph_attrs),
            method=("method", [method_used], method_attrs),
        )

        if method_used == "fit":
            metric_maxs = xr.concat(metric_output[nhems:], dim="hemsph")  # type: ignore
            metric_maxs.attrs = {
                "long_name": self.metric_name.upper() + " metric max",
                "unit": self._obj.attrs.get("unit", "m.s-1"),
            }
            metric_maxs = metric_maxs.expand_dims("method").assign_coords(
                hemsph=("hemsph", hemsph, hemsph_attrs),
                method=("method", [method_used], method_attrs),
            )

            return metric_lats, metric_maxs  # type: ignore
        else:
            return metric_lats

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
            "z": ["z", "h", "height", "geopotentialheight"],
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
        self.xarray_data = xarray_data

        self.lat_name = self.extract_property_name(property_name=["lat", "Latitude"])
        latitudes: np.ndarray = xarray_data[self.lat_name].values
        self.latitudes = latitudes
        self.params["lat"] = latitudes

        try:
            self.pres_name = self.extract_property_name(
                property_name=["pres", "Pressure"]
            )
        except KeyError as e:
            if self.require_pres_axis:
                raise e

        # check for pressure if required or present for optional ones
        if self.require_pres_axis or (
            self.metric_name in ["edj", "uas"]
            and (self.pres_name in self.xarray_data.dims)
        ):
            pressure: np.ndarray = xarray_data[self.pres_name].values
            self.params["lev"] = pressure
        # otherwise don't use it if it is present
        else:
            self.pres_name = ""

        if self.metric_name == "tpb":
            if self.params.get("method", "") == "cutoff":
                z_name = self.extract_property_name(
                    property_name=["z", "Geopotential height"]
                )
                self.params["Z"] = self._obj[z_name]
            else:
                self.params.pop("Z", None)

        return

    def xr_edj(self, **params) -> xr.DataArray:

        """
        TropD Eddy Driven Jet (EDJ) Metric

        Finds the latitude of the maximum of zonal wind at the level closest to 850 hPa

        The :py:class:`Dataset` should contain one variable corresponding to zonal wind.
        If multiple non-coordinate variables are in the dataset, this method attempts to
        guess which field corresponds to zonal wind based on field's name. The
        :py:class:`Dataset` should also contain a latitude-like dimension. If a
        pressure-like dimension is included, level closest to 850hPa is chosen

        Parameters
        ----------
        method : {"peak", "max", "fit"}, optional
            Method for determining latitude of maximum zonal wind, by default "peak":

            * "peak": Latitude of the maximum of the zonal wind at the level closest
                      to 850hPa (smoothing parameter ``n=30``)
            * "max": Latitude of the maximum of the zonal wind at the level closest to
                     850hPa (smoothing parameter ``n=6``)
            * "fit": Latitude of the maximum of the zonal wind at the level closest to
                     850hPa using a quadratic polynomial fit of data from grid points
                     surrounding the grid point of the maximum

        n_fit : int, optional
            used when ``method="fit"``, determines the number of points around the max to
            use while fitting, by default 1

        **kwargs : optional
            additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
            for ``method="fit"``)

            n : int, optional
                Rank of moment used to calculate the location of max, e.g.,
                ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, by default 30 if
                ``method="peak"``

        Returns
        -------
        EDJ_metrics: xarray.DataArray
            :py:class:`DataArray` of EDJ metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)
        EDJ_max : xarray.DataArray, conditional
            :py:class:`DataArray` of EDJ metric maxima with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes), returned if ``method="fit"``

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like dimension cannot be identified in the :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            zonal wind cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> ds = xr.open_dataset("pytropd/ValidationData/ua.nc")
        >>> EDJ_metrics = ds.pyt_metrics.xr_edj(method="max")
        >>> EDJ_metrics.sel(hemsph="NH").plot()
        >>> plt.show()
        """

        self.metric_name = "edj"
        self.params = params
        return self.compute_metrics()

    def xr_olr(self, **params) -> xr.DataArray:

        """
        TropD Outgoing Longwave Radiation (OLR) Metric

        Finds the latitude of maximum OLR or first latitude poleward of maximum where OLR
        reaches crosses a predefined cutoff.

        The :py:class:`Dataset` should contain one variable corresponding to outgoing
        longwave radiation at TOA (positive upward) in :math:`W/m^2`. If multiple
        non-coordinate variables are in the dataset, this method attempts to guess which
        field corresponds to OLR based on field's name. The :py:class:`Dataset` should
        also contain a latitude-like dimension. If a pressure-like dimension is included,
        the level closest to 850hPa is chosen

        Parameters
        ----------
        method : {"250W", "20W", "cutoff", "10Perc", "max", "peak"}, optional
            Method for determining the OLR maximum/threshold, by default "250W":

            * "250W": the 1st latitude poleward of the tropical OLR max in each hemisphere
                      where OLR crosses :math:`250W/m^2`
            * "20W": the 1st latitude poleward of the tropical OLR max in each hemisphere
                     where OLR crosses [the tropical OLR max minus :math:`20W/m^2`]
            * "cutoff": the 1st latitude poleward of the tropical OLR max in each
                        hemisphere where OLR crosses a specified cutoff value
            * "10Perc": the 1st latitude poleward of the tropical OLR max in each
                        hemisphere where OLR is 10% smaller than the tropical OLR max
            * "max": the latitude of the tropical olr max in each hemisphere with
                     smoothing parameter ``n=6``
            * "peak": the latitude of maximum of tropical olr in each hemisphere with
                      smoothing parameter ``n=30``

        Cutoff : float, optional
            if ``method="cutoff"``, specifies the OLR cutoff value in :math:`W/m^2`

        **kwargs : optional
            additional keyword arguments for :py:func:`TropD_Calculate_MaxLat`

            n : int, optional
                Rank of moment used to calculate the location of max, e.g.,
                ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if
                ``method="peak"``

        Returns
        -------
        OLR_metrics: xarray.DataArray
            :py:class:`DataArray` of OLR metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like dimension cannot be identified in the :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            outgoing longwave radiation cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> ds = -xr.open_dataset("pytropd/ValidationData/rlnt.nc")
        >>> OLR_metrics = ds.pyt_metrics.xr_olr(method="cutoff", Cutoff=220.0)
        >>> OLR_metrics.sel(hemsph="NH").plot()
        >>> plt.show()
        """

        self.metric_name = "olr"
        self.params = params
        self.require_pres_axis = False
        return self.compute_metrics()

    def xr_pe(self, **params) -> xr.DataArray:

        """
        TropD Precipitation Minus Evaporation (PE) Metric

        Finds the first zero crossing of zonal-mean precipitation minus evaporation
        poleward of the subtropical minimum

        The :py:class:`Dataset` should contain one variable corresponding to P-E.
        If multiple non-coordinate variables are in the dataset, this method attempts to
        guess which field corresponds to P-E based on field's name. The
        :py:class:`Dataset` should also contain a latitude-like dimension. If a
        pressure-like dimension is included, level closest to 850hPa is chosen

        Parameters
        ----------
        method : {"zero_crossing"}, optional
            Method to compute the zero crossing for precipitation minus evaporation, by
            default "zero_crossing":
            * "zero_crossing": the first latitude poleward of the subtropical P-E min
                               where P-E changes from negative to positive.

        lat_uncertainty : float, optional
            The minimal distance allowed between adjacent zero crossings in degrees,
            by default 0.0

        Returns
        -------
        PE_metrics: xarray.DataArray
            :py:class:`DataArray` of PE metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like dimension cannot be identified in the :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            precipitation minus evaporation cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> prds = xr.open_dataset("pytropd/ValidationData/pr.nc")
        >>> eds = -xr.open_dataset("pytropd/ValidationData/hfls.nc") / 2510400.0
        >>> ds = (prds.pr - eds.hfls).to_dataset(name="pe")
        >>> ds.pyt_metrics.xr_pe().sel(hemsph="NH").plot()
        >>> plt.show()
        """

        self.metric_name = "pe"
        self.params = params
        self.require_pres_axis = False
        return self.compute_metrics()

    def xr_psi(self, **params) -> xr.DataArray:

        """
        TropD Mass Streamfunction (PSI) Metric

        Finds the latitude of the subtropical zero crossing of the meridional mass
        streamfunction

        The :py:class:`Dataset` should contain one variable corresponding to meridional
        wind. If multiple non-coordinate variables are in the dataset, this method
        attempts to guess which field corresponds to meridional wind based on field's
        name. The :py:class:`Dataset` should also contain a latitude-like dimension and a
        pressure-like dimension

        Parameters
        ----------
        method : {"Psi_500", "Psi_500_10Perc", "Psi_300_700", "Psi_500_Int", "Psi_Int"},
        optional
            Method of determining which Psi zero crossing to return, by default "Psi_500":

            * "Psi_500": Zero crossing of the streamfunction (Psi) at 500hPa
            * "Psi_500_10Perc": Crossing of 10% of the extremum value of Psi in each
                                hemisphere at the 500hPa level
            * "Psi_300_700": Zero crossing of Psi vertically averaged between the 300hPa
                             and 700 hPa levels
            * "Psi_500_Int": Zero crossing of the vertically-integrated Psi at 500 hPa
            * "Psi_Int" : Zero crossing of the column-averaged Psi

        lat_uncertainty : float, optional
            The minimal distance allowed between the adjacent zero crossings, same units
            as lat, by default 0.0. e.g., for ``lat_uncertainty=10``, this function will
            return NaN if another zero crossing is within 10 degrees of the most
            equatorward zero crossing.

        Returns
        -------
        PSI_metrics: xarray.DataArray
            :py:class:`DataArray` of PSI metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)

        Raises
        ------
        :py:class:`KeyError`:
            If latitude-like and pressure-like dimensions cannot be identified in the
            :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the :py:class:`Dataset` and a variable
            corresponding to meridional wind cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> ds = xr.open_dataset("pytropd/ValidationData/va.nc")
        >>> PSI_metrics = ds.pyt_metrics.xr_psi(method="Psi_500_10Perc")
        >>> PSI_metrics.squeeze().plot.line(x="time", add_legend=False)
        >>> plt.show()
        """

        self.metric_name = "psi"
        self.params = params
        self.require_pres_axis = True
        return self.compute_metrics()

    def xr_psl(self, **params) -> xr.DataArray:

        """
        TropD Sea-level Pressure (PSL) Metric

        Finds the latitude of maximum of the subtropical sea-level pressure

        The :py:class:`Dataset` should contain one variable corresponding to sea-level
        pressure. If multiple non-coordinate variables are in the dataset, this method
        attempts to guess which field corresponds to sea-level pressure based on field's
        name. The :py:class:`Dataset` should also contain a latitude-like dimension

        Parameters
        ----------
        method : {"peak", "max"}, optional
            Method for determining latitude of max PSL, by default "peak":

            * "peak": latitude of the maximum of subtropical sea-level pressure (smoothing
                    parameter ``n=30``)
            * "max": latitude of the maximum of subtropical sea-level pressure (smoothing
                    parameter ``n=6``)

        **kwargs : optional
            additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
            for ``method="fit"``)

            n : int, optional
                Rank of moment used to calculate the location of max, e.g.,
                ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if
                ``method="peak"``
        Returns
        -------
        PSL_metrics: xarray.DataArray
            :py:class:`DataArray` of PSL metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like dimension cannot be identified in the :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            sea-level pressure cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> from pandas import date_range
        >>> pslds = xr.open_dataset("pytropd/ValidationData/psl.nc")
        >>> time_index = date_range(
        ...     start="1979-01-01", periods=pslds.time.size, freq="MS"
        ... )
        >>> pslds = pslds.assign_coords(time = time_index)
        >>> ds = pslds.resample(time="QS-DEC").mean()
        >>> PSL_metrics = ds.pyt_metrics.xr_psl(method="max").sel(hemsph="NH").squeeze()
        >>> PSL_seasonal_metrics_list = [
        ...     PSL_metrics.isel(time=group).expand_dims(season=[ssn])
        ...     for ssn, group in PSL_metrics.groupby('time.season').groups.items()
        ... ]
        >>> PSL_seasonal_metrics = xr.concat(
        ...     [
        ...         da.assign_coords(time=da.time.dt.year).rename(time="year")
        ...         for da in PSL_seasonal_metrics_list
        ...     ],
        ...     dim='season',
        ... )
        >>> PSL_seasonal_metrics.plot.line(x="year")
        >>> plt.show()
        """

        self.metric_name = "psl"
        self.params = params
        self.require_pres_axis = False
        return self.compute_metrics()

    def xr_stj(self, **params) -> xr.DataArray:

        """
        TropD Subtropical Jet (STJ) Metric

        Finds the latitude of the (adjusted) maximum upper-level zonal-mean zonal wind

        The :py:class:`Dataset` should contain one variable corresponding to zonal wind.
        If multiple non-coordinate variables are in the dataset, this method attempts to
        guess which field corresponds to zonal wind based on field's name. The
        :py:class:`Dataset` should also contain a latitude-like dimension and a
        pressure-like dimension.

        Parameters
        ----------
        method : {"adjusted_peak", "adjusted_max", "core_peak", "core_max"}, optional
            Method for determing the latitude of the STJ maximum, by default
            "adjusted_peak":

            * "adjusted_peak": Latitude of maximum (smoothing parameter``n=30``) of [the
                               zonal wind averaged between 100 and 400 hPa] minus [the
                               zonal mean zonal wind at the level closest to 850hPa],
                               poleward of 10 degrees and equatorward of the Eddy Driven
                               Jet latitude
            * "adjusted_max": Latitude of maximum (smoothing parameter ``n=6``) of [the
                              zonal wind averaged between 100 and 400 hPa] minus [the
                              zonal mean zonal wind at the level closest to 850hPa],
                              poleward of 10 degrees and equatorward of the Eddy Driven
                              Jet latitude
            * "core_peak": Latitude of maximum (smoothing parameter ``n=30``) of the zonal
                           wind averaged between 100 and 400 hPa, poleward of 10 degrees
                           and equatorward of 70 degrees
            * "core_max": Latitude of maximum (smoothing parameter ``n=6``) of the zonal
                          wind averaged between 100 and 400 hPa, poleward of 10 degrees
                          and equatorward of 70 degrees
            * "fit": Latitude of the maximum of [the zonal wind averaged between 100 and
                     400 hPa] minus [the zonal mean zonal wind at the level closest to
                     850hPa] using a quadratic polynomial fit of data from grid points
                     surrounding the grid point of the maximum

        **kwargs : optional
            additional keyword arguments for :py:func:`TropD_Calculate_MaxLat`

            n : int, optional
                Rank of moment used to calculate the location of max, e.g.,
                ``n=1,2,4,6,8,...``, by default 6 if ``method="core_max"`` or
                ``method="adjusted_max"``, 30 if ``method="core_peak"`` or
                ``method="adjusted_peak"``

        Returns
        -------
        STJ_metrics: xarray.DataArray
            :py:class:`DataArray` of STJ metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)
        STJ_max : xarray.DataArray, conditional
            :py:class:`DataArray` of STJ metric maxima with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes), returned if ``method="fit"``

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like or pressure-like dimension cannot be identified in the
            :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            zonal wind cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> ds = xr.open_dataset("pytropd/ValidationData/ua.nc")
        >>> time_index = date_range(
        ...     start="1979-01-01", periods=pslds.time.size, freq="MS"
        ... )
        >>> ds = ds.assign_coords(time=time_index).resample(time='AS').mean()
        >>> STJ_core = ds.pyt_metrics.xr_stj(method="core_max")
        >>> STJ_adjusted = ds.pyt_metrics.xr_stj(method="adjusted_max")
        >>> STJ_metrics = xr.concat([STJ_core,STJ_adjusted], dim='method')
        >>> STJ_metrics.sel(hemsph="NH").plot.line(x='time')
        >>> plt.show()
        """

        self.metric_name = "stj"
        self.params = params
        self.require_pres_axis = True
        return self.compute_metrics()

    def xr_tpb(self, **params) -> xr.DataArray:
        """
        TropD Tropopause Break (TPB) Metric

        Finds the latitude of the tropopause break

        The :py:class:`Dataset` should contain one variable corresponding to temperature.
        If multiple non-coordinate variables are in the dataset, this method attempts to
        guess which field corresponds to temperature based on field's name. The
        :py:class:`Dataset` should also contain a latitude-like dimension and a
        pressure-like dimension. If using ``method="cutoff"``, there should be an
        additional variable in the :py:class:`Dataset` corresponding to geopotential
        height in meters

        Parameters
        ----------
        method : {"max_gradient", "cutoff", "max_potemp"}, optional
            Method to identify tropopause break, by default "max_gradient":

            * "max_gradient": The latitude of maximal poleward gradient of the tropopause
                              height

            * "cutoff": The most equatorward latitude where the tropopause crosses a
                        prescribed cutoff value

            * "max_potemp": The latitude of maximal difference between the potential
                            temperature at the tropopause and at the surface

        Cutoff : float, optional
            Geopotential height cutoff (m) that marks the location of the tropopause
            break, by default 1.5e4, used when ``method="cutoff"``
        **kwargs : optional
            additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
            for ``method="cutoff"``)

        Returns
        -------
        TPB_metrics: xarray.DataArray
            :py:class:`DataArray` of TPB metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like or a pressure-like dimension cannot be identified in the
            :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            temperature cannot be detected

        :py:class:`KeyError`:
            If ``method="cutoff"`` and a variable corresponding to geopotential height
            cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> tds = xr.open_dataset("pytropd/ValidationData/ta.nc").rename(ta="T")
        >>> zds = xr.open_dataset("pytropd/ValidationData/zg.nc").rename(zg="Z")
        >>> ds = xr.merge([tds, zds])
        >>> TPB_metrics = ds.pyt_metrics.xr_tpb(method="cutoff")
        >>> TPB_metrics.sel(hemsph="NH").plot()
        >>> plt.show()
        """

        self.metric_name = "tpb"
        self.params = params
        self.require_pres_axis = True
        return self.compute_metrics()

    def xr_uas(self, **params) -> xr.DataArray:
        """
        TropD Near-Surface Zonal Wind (UAS) Metric

        The :py:class:`Dataset` should contain one variable corresponding to zonal wind.
        If multiple non-coordinate variables are in the dataset, this method attempts to
        guess which field corresponds to zonal wind based on field's name. The
        :py:class:`Dataset` should also contain a latitude-like dimension. If a
        pressure-like dimension is included, level closest to 850hPa is chosen

        Parameters
        ----------
        method : {"zero_crossing"}, optional
            Method for identifying the surface wind zero crossing, by default
            "zero_crossing":

            * "zero_crossing": the first subtropical latitude where near-surface zonal
                               wind changes from negative to positive

        lat_uncertainty : float, optional
            the minimal distance allowed between adjacent zero crossings in degrees, by
            default 0.0

        Returns
        -------
        UAS_metrics: xarray.DataArray
            :py:class:`DataArray` of UAS metric latitudes with new dimensions
            ``hemsph`` (SH latitudes, NH latitudes)

        Raises
        ------
        :py:class:`KeyError`:
            If a latitude-like dimension cannot be identified in the :py:class:`Dataset`

        :py:class:`KeyError`:
            If multiple variables are in the dataset and a variable corresponding to
            zonal wind cannot be detected

        Examples
        --------
        .. code-block:: python
        >>> import xarray as xr
        >>> import pytropd as pyt
        >>> import matplotlib.pyplot as plt
        >>> ds = xr.open_dataset("pytropd/ValidationData/ua.nc")
        >>> ds.pyt_metrics.xr_uas().sel(hemsph="NH").plot()
        >>> plt.show()
        """

        self.metric_name = "uas"
        self.params = params
        return self.compute_metrics()
