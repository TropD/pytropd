# Written by Ori Adam Mar.21.2017
# Edited by Alison Ming Jul.4.2017
# rewrite for readability/vectorization - sjs 1.27.22
from typing import Optional, Tuple, Callable, Union
import warnings
from functools import wraps
from inspect import signature
import numpy as np
from numpy.polynomial import polynomial
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from functions import (
#from .functions import (
    find_nearest,
    TropD_Calculate_MaxLat,
    TropD_Calculate_MaxPres,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_TropopauseHeight,
    TropD_Calculate_ZeroCrossing,
)
from stratospheric_metrics import (
#from .stratospheric_metrics import (
    TropD_Metric_CL,
    TropD_Metric_TAL,
    TropD_Metric_GWL,
    TropD_Metric_ONESIGMA,
)

# kappa = R_dry / Cp_dry
KAPPA = 287.04 / 1005.7


def hemisphere_handler(
    metric_func: Callable[..., Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
) -> Callable:
    """
    Wrapper for metrics to allow one or two hemispheres of data to be passed

    Parameters
    ----------
    metric_func : Callable
        the pytropd metric function to wrap

    Returns
    -------
    Callable
        wrapped function with same name, docstring, and call signature as the original
        function, but which operates over each hemisphere independently
    """

    @wraps(metric_func)
    def wrapped_metric_func(
        arr: np.ndarray, 
        lat: np.ndarray, 
        *args, 
        **kwargs
    ) -> dict[str, np.ndarray]:
        """
        wrapper for pytropd metric functions which allows one or two hemispheres of data
        to be passed

        Parameters
        ----------
        arr : np.ndarray
            first argument to a pytropd metric function is always the field used for the
            metric (zonal wind, OLR, meridional wind, etc.)
        lat : np.ndarray
            second argument to a metric function is always the latitude array
        *args
            Other arguments passed to the metric function, typically will only be the
            pressure level array if any
        **kwargs
            Keyword arguments passed to the metric function, all have a `method` keyword,
            some have others

        Returns
        -------
        PhiSH : np.ndarray, optional
            metric latitude in SH (if SH latitudes are provided)
        PhiNH : np.ndarray, optional
            metric latitude in NH (if NH latitudes are provided)
        UmaxSH : np.ndarray, optional
            maximum of SH zonal wind (if using `TropD_Metric_EDJ` or `TropD_Metric_STJ`
            and `method="fit"`)
        UmaxNH : np.ndarray, optional
            maximum of NH zonal wind (if using `TropD_Metric_EDJ` or `TropD_Metric_STJ`
            and `method="fit"`)
        """
        #We need to package things up as a tuple rather than a dict for xarray to be happy
        returning_to_xarray_bool = kwargs.pop('xarray_bool',False)
        list_for_xarray = []

        kwargs.pop("hem", None)
        metric_value_bool = kwargs.get('metric_value', False)
        ## Do we need the vertical position of the metric? STJ only at the moment
        #vertical_position_bool = kwargs.get('vertical_position', False)



        # get the metric identifier from the function name
        metric_code = metric_func.__name__.split("_")[-1]
        # we need to know which dimension is the latitude dimension to split it
        # appropriately. In functions which accept vertically-resolved input data,
        # the vertical level is last and latitude is second to last. Otherwise, latitude
        # is last
        has_extra_dim = False
        # these all require vertically-resolved data
        if metric_code in ["STJ", "STJV", "TPB", "PSI"]:
            has_extra_dim = True
        # these take it optionally, so we need to figure out if it was provided, either as
        # an arg (it will be third positional and probably the only arg in args) or kwarg
        elif metric_code in ["UAS", "EDJ"]:
            has_extra_dim = (len(args) > 0) or (kwargs.get("lev", None) is not None)
        # Shah metrics may also have an extra last dimension if zonal_mean_tracer=False
        # (but it is lon, not lev)
        elif metric_code in ["GWL", "1sigma"]:
            has_extra_dim = not kwargs.get(
                "zonal_mean_tracer",
                signature(metric_func).parameters["zonal_mean_tracer"].default,
            )
        # now the TropD_Metric_TPB accepts a Z array kwarg that also needs to be split
        # based on latitude, so we need to get that if it is there
        Z = None
        if metric_code == "TPB":
            Z = kwargs.pop("Z", None)
        # now we just split by hemisphere and apply as if the data was for the NH
        output_dict = {}
        if (lat < -20.0).any():
            # let's make sure we include the equator point just in case
            SHmask = lat < 0.5
            SHarr_mask = [Ellipsis, SHmask]
            if has_extra_dim:
                SHarr_mask.append(slice(None))
            if Z is not None:
                kwargs["Z"] = Z[tuple(SHarr_mask)]

            metric_func_output = metric_func(
                    # for the TropD_Metric_PSI, it takes meridional wind. In order to
                    # make meridional wind in the SH like the NH, we have to flip the sign
                    arr[tuple(SHarr_mask)] * (-1.0 if metric_code == "PSI" else 1.0),
                    -lat[SHmask],
                    *args,
                    **kwargs,
                )
            output_dict['phiSH'] = -metric_func_output['phi']
            if metric_code=="STJV": 
                output_dict['levelSH'] = metric_func_output['level']
                
            if metric_value_bool:
                output_dict['metric_valueSH'] = metric_func_output['metric_value'] * (-1.0 if metric_code == "PSI" else 1.0)
            
            #if vertical_position_bool:
            #    output_dict['vertical_positionSH'] = metric_func_output['vertical_position'] 


        if (lat > 20.0).any():
            NHmask = lat > -0.5
            NHarr_mask = [Ellipsis, NHmask]
            if has_extra_dim:
                NHarr_mask.append(slice(None))
            if Z is not None:
                kwargs["Z"] = Z[tuple(NHarr_mask)]
            metric_func_output = metric_func(arr[tuple(NHarr_mask)], lat[NHmask], *args, **kwargs)
            output_dict['phiNH'] = metric_func_output['phi']
            if metric_code=="STJV": 
                output_dict['levelNH'] = metric_func_output['level']

            if metric_value_bool:
                output_dict['metric_valueNH'] = metric_func_output['metric_value']               
            #if vertical_position_bool:
            #    output_dict['vertical_positionNH'] = metric_func_output['vertical_position'] 

        if returning_to_xarray_bool:
           if (lat < -20.0).any(): list_for_xarray.append(output_dict['phiSH'])
           if (lat > 20.0).any(): list_for_xarray.append(output_dict['phiNH'])
           if metric_value_bool:
               if (lat < -20.0).any(): list_for_xarray.append(output_dict['metric_valueSH'])
               if (lat > 20.0).any(): list_for_xarray.append(output_dict['metric_valueNH'])
           if metric_code=="STJV": 
               list_for_xarray.append(output_dict['levelSH'])
               list_for_xarray.append(output_dict['levelNH'])
           return tuple(list_for_xarray)
        else:
           return output_dict

    return wrapped_metric_func


@hemisphere_handler
def TropD_Metric_EDJ(
    U: np.ndarray,
    lat: np.ndarray,
    lev: Optional[np.ndarray] = None,
    method: str = "peak",
    n_fit: int = 1,
    **maxlat_kwargs,
) -> dict[str,np.ndarray]:
    """
    TropD Eddy Driven Jet (EDJ) Metric

    Latitude of maximum of the zonal wind at the level closest to 850 hPa

    Parameters
    ----------
    U : numpy.ndarray (dim1, ..., lat[, lev])
        N-dimensional array of zonal-mean zonal wind data. Also accepts surface/
        850hPa wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray, optional (lev,)
        vertical level array in hPa, used to find wind closest to 850hPa. if not
        provided, last axis is assumed to be lat
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
        used when ``method="fit"``, determines the number of points around the max to use
        while fitting, by default 1
    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="fit"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if ``method="peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional latitudes of the EDJ
    Umax : numpy.ndarray (dim1, ..., dimN-2), conditional
        N-2 dimensional EDJ strength, returned if ``method="fit"``

    """

    U = np.asarray(U)
    lat = np.asarray(lat)
    get_lev = lev is not None
    U_grid_shape = U.shape[-2:] if get_lev else U.shape[-1]
    input_grid_shape = (lat.size, lev.size) if get_lev else lat.size  # type: ignore
    if U_grid_shape != input_grid_shape:
        raise ValueError(
            f"last axes of U w/ shape {U_grid_shape},"
            " not aligned with input grid of shape " + str(input_grid_shape)
        )
    if method not in ["max", "peak", "fit"]:
        raise ValueError("unrecognized method " + method)
    n_fit = int(n_fit)

    eq_boundary = 15.0
    polar_boundary = 70.0
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    metric_value_bool = maxlat_kwargs.pop('metric_value', False)

    if get_lev:
        u850 = U[..., find_nearest(lev, 850.0)]  # type: ignore
    else:
        u850 = U

    if method != "fit":  # max or peak
        # update default n to 30 for peak
        if method == "peak":
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
        else:  # max
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 6)
        # lat should already be last axis
        maxlat_kwargs.pop("axis", None)

        Phi = TropD_Calculate_MaxLat(u850[..., mask], lat[mask], **maxlat_kwargs)

        i_lat_max = [np.abs(lat-Phi.flat[i]).argmin() for i in range(np.size(Phi))] 
        u_flat = u850.reshape(-1, lat.size)
        
        Umax = np.zeros(u850.shape[:-1])
        for i in range(len(i_lat_max)):
            phi_ind = np.unravel_index(i, u850.shape[:-1])
            Umax[phi_ind] = u_flat[i,i_lat_max[i]]


    else:  # method == 'fit':
        u_flat = u850.reshape(-1, lat.size)

        lat_mask = lat[mask]
        Phi = np.zeros(u850.shape[:-1])
        Umax = np.zeros(u850.shape[:-1])
        for i, Uh in enumerate(u_flat[:, mask]):
            Im = np.nanargmax(Uh)
            phi_ind = np.unravel_index(i, u850.shape[:-1])

            if Im == 0 or Im == Uh.size - 1:
                Phi[phi_ind] = lat_mask[Im]
                continue

            if n_fit > Im or n_fit > Uh.size - Im + 1:
                N = np.min(Im, Uh.size - Im + 1)
            else:
                N = n_fit
            p = polynomial.polyfit(
                lat_mask[Im - N : Im + N + 1], Uh[Im - N : Im + N + 1], deg=2
            )
            # vertex of quadratic ax**2+bx+c is at -b/2a
            # p[0] + p[1]*x + p[2]*x**2
            Phi[phi_ind] = -p[1] / (2.0 * p[2])
            # value at vertex is (4ac-b**2)/(4a)
            Umax[phi_ind] = (4.0 * p[2] * p[0] - p[1] * p[1]) / 4.0 / p[2]

    if metric_value_bool:
        return dict(phi=Phi, metric_value=Umax)
    else:
        return dict(phi=Phi)


@hemisphere_handler
def TropD_Metric_OLR(
    olr: np.ndarray,
    lat: np.ndarray,
    method: str = "250W",
    Cutoff: Optional[float] = None,
    **maxlat_kwargs,
) -> dict[str,np.ndarray]:
    """
    TropD Outgoing Longwave Radiation (OLR) Metric

    Latitude of maximum OLR or first latitude poleward of maximum where OLR reaches
    crosses a predefined cutoff

    Parameters
    ----------
    olr : numpy.ndarray (dim1, ..., lat)
        zonal mean TOA olr (positive)
    lat : numpy.ndarray (lat,)
        latitude array
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
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if ``method="peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional latitudes of the near-equator OLR max/threshold crossing
    """

    olr = np.asarray(olr)
    lat = np.asarray(lat)
    metric_value_bool = maxlat_kwargs.pop('metric_value', False)

    if olr.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of olr had shape {olr.shape[-1]},"
            f" not aligned with lat size {lat.size}"
        )
    if "n" in maxlat_kwargs and method != "max":
        warnings.warn("smoothing parameter n only utilized for method == max")
        maxlat_kwargs.pop("n")
    if Cutoff is not None and method != "cutoff":
        warnings.warn("cutoff parameter only utilized for method == cutoff")

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        olr = olr[..., ::-1]
        lat = lat[::-1]

    eq_boundary = 5.0
    subpolar_boundary = 40.0
    polar_boundary = 60.0

    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)

    if method == "peak":
        maxlat_kwargs["n"] = 30
    elif method == "max":
        maxlat_kwargs["n"] = 6
    # lat should already be last axis
    maxlat_kwargs.pop("axis", None)

    olr_max_lat = TropD_Calculate_MaxLat(
        olr[..., subpolar_mask], lat[subpolar_mask], **maxlat_kwargs
    )

    if method in ["cutoff", "250W", "20W", "10Perc"]:
        # get tropical OLR max for methods 20W and 10Perc
        olr_max = olr[..., subpolar_mask].max(axis=-1)[..., None]

        # set cutoff dependent on method
        if method == "250W":
            Cutoff = 250.0
        elif method == "20W":
            Cutoff = olr_max - 20.0
        elif method == "10Perc":
            Cutoff = 0.9 * olr_max

        # identify regions poleward of the OLR max in both hemispheres
        mask = (lat > eq_boundary) & (lat < polar_boundary)
        max_mask = lat[mask] > olr_max_lat[..., None]

        # OLR in each hemisphere, only valid poleward of max
        olr = np.where(max_mask, olr[..., mask], np.nan)

        # get latitude where OLR falls below cutoff poleward of tropical max
        Phi = TropD_Calculate_ZeroCrossing(olr - Cutoff, lat[mask])
    # these methods don't need to find a threshold after
    # the OLR max, just need the max
    elif method in ["max", "peak"]:
        Phi = olr_max_lat
    else:
        raise ValueError("unrecognized method " + method)


    if metric_value_bool:
        olrmax = np.zeros(olr.shape[:-1])
        olr_flat = olr.reshape(-1, lat.size)
        i_lat_max = [np.abs(lat-Phi.flat[i]).argmin() for i in range(np.size(Phi))] 

        for i in range(len(i_lat_max)):
            phi_ind = np.unravel_index(i, olr.shape[:-1])
            olrmax[phi_ind] = olr_flat[i,i_lat_max[i]]
        return dict(phi=Phi, metric_value=olrmax)
    else:
        return dict(phi=Phi)


@hemisphere_handler
def TropD_Metric_PE(
    pe: np.ndarray,
    lat: np.ndarray,
    method: str = "zero_crossing",
    lat_uncertainty: float = 0.0,
) -> dict[str,np.ndarray]:
    """
    TropD Precipitation Minus Evaporation (PE) Metric

    Find the first zero crossing of zonal-mean precipitation minus evaporation poleward
    of the subtropical minimum

    Parameters
    ----------
    pe : numpy.ndarray (dim1,  ..., lat,)
        zonal-mean precipitation minus evaporation
    lat : numpy.ndarray (lat,)
        latitude array
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
    Phi : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional latitudes of the 1st subtropical P-E zero crossing
    """

    pe = np.atleast_2d(pe)
    lat = np.asarray(lat)

    if pe.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of P-E had shape {pe.shape[-1]},"
            f" not aligned with lat size {lat.size}"
        )
    if method != "zero_crossing":
        raise ValueError("unrecognized method " + method)

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        pe = pe[..., ::-1]
        lat = lat[::-1]

    # The gradient of PE is used to determine whether PE
    # becomes positive at the zero crossing
    dpedy = np.diff(pe, axis=-1)
    lat_mid = (lat[:-1] + lat[1:]) / 2.0
    # interpolate back to original grid, duplicating
    # boundary behavior of np.interp
    pe_grad = interp1d(
        lat_mid,
        dpedy,
        axis=-1,
        bounds_error=False,
        fill_value=(dpedy[..., 0], dpedy[..., -1]),
    )(lat)

    # define latitudes of boundaries certain regions
    eq_boundary = 5.0
    subpolar_boundary = 50.0
    polar_boundary = 60.0

    # split into hemispheres
    mask = (lat > eq_boundary) & (lat < polar_boundary)
    lat_masked = lat[mask]

    # find E-P maximum (P-E min) latitude in subtropics
    # first define the subpolar region to search, excluding poles due to low P-E
    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    Emax_lat_masked = TropD_Calculate_MaxLat(
        -pe[..., subpolar_mask], lat[subpolar_mask], n=30
    )

    # find zero crossings poleward of E-P max
    # first define regions poleward of E-P max in each hemisphere
    lat_after_Emax = lat_masked > Emax_lat_masked[..., None]
    pelat_after_Emax = np.where(lat_after_Emax, pe[..., mask], np.nan)
    ZC1_lat_masked = TropD_Calculate_ZeroCrossing(
        pelat_after_Emax, lat_masked, lat_uncertainty=lat_uncertainty
    )

    # we've got the zero crossing poleward of E-P max, but it might not be the
    # right one. Now we need to go through and check the P-E gradient
    # to make sure P-E is increasing poleward.
    # if it is, use that latitude, else, use the next zero crossing

    # first check if the (northward) gradient value at the
    # zero crossing is increasing poleward
    pelat_increases_at_ZC = np.zeros_like(ZC1_lat_masked)
    pe_grad_flat = pe_grad.reshape(-1, lat.size)
    for i, ipe_grad in enumerate(pe_grad_flat):
        i_unrav = np.unravel_index(i, pe_grad.shape[:-1])
        interp_pe_grad = interp1d(lat, ipe_grad, axis=-1)
        pelat_increases_at_ZC[i_unrav] = interp_pe_grad(ZC1_lat_masked.flatten()[i]) > 0

    # then get the next zero crossing for when we need it
    # first define regions poleward of zero crossing
    lat_after_ZC = lat_masked > ZC1_lat_masked[..., None]
    pelat_after_ZC = np.where(lat_after_ZC, pe[..., mask], np.nan)
    ZC2_lat_masked = TropD_Calculate_ZeroCrossing(
        pelat_after_ZC, lat_masked, lat_uncertainty=lat_uncertainty
    )

    # if the gradient is increasing poleward, use it, otherwise, use the next
    Phi = np.where(pelat_increases_at_ZC, ZC1_lat_masked, ZC2_lat_masked)

    return dict(phi=Phi)



@hemisphere_handler
def TropD_Metric_PSI(
    V: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "Psi_500",
    lat_uncertainty: float = 0.0,
) -> dict[str,np.ndarray]:
    """
    TropD Mass Streamfunction (PSI) Metric

    Latitude of the subtropical zero crossing of the meridional mass streamfunction

    Parameters
    ----------
    V : numpy.ndarray (dim1, ..., lat, lev)
        N-dimensional zonal-mean meridional wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical level array in hPa
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
        The minimal distance allowed between the adjacent zero crossings, same units as
        lat, by default 0.0. e.g., for ``lat_uncertainty=10``, this function will return
        NaN if another zero crossing is within 10 degrees of the most equatorward
        zero crossing.

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the Psi zero crossing
    """

    # type casting/input checking
    V = np.asarray(V)
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if V.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"final dimensions on V {V.shape[-2:]} and grid "
            f"coordinates don't match ({lat.size},{lev.size})"
        )

    Psi = TropD_Calculate_StreamFunction(V, lat, lev)

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        Psi = Psi[..., ::-1, :]
        lat = lat[::-1]
    cos_lat = np.cos(lat * np.pi / 180.0)[:, None]

    if (method == "Psi_500") or (method == "Psi_500_10Perc"):
        # Use Psi at the level nearest to 500 hPa
        P = Psi[..., find_nearest(lev, 500.0)]
    elif method == "Psi_300_700":
        # Use Psi averaged between the 300 and 700 hPa level
        layer_700_to_300 = (lev <= 700.0) & (lev >= 300.0)
        P = np.trapz(
            Psi[..., layer_700_to_300] * cos_lat, lev[layer_700_to_300] * 100.0, axis=-1
        )
    elif method == "Psi_500_Int":
        # Use integrated Psi from p=0 to level nearest to 500 hPa
        PPsi = cumtrapz(Psi * cos_lat, 100.0 * lev, axis=-1, initial=0.0)
        P = PPsi[..., find_nearest(lev, 500.0)]
    elif method == "Psi_Int":
        # Use vertical mean of Psi
        P = np.trapz(Psi * cos_lat, 100.0 * lev, axis=-1)
    else:
        raise ValueError("unrecognized method ", method)

    # define regions of interest
    subpolar_boundary = 30.0
    polar_boundary = 60.0
    mask = (lat > 0) & (lat < polar_boundary)
    subpolar_mask = (lat > 0) & (lat < subpolar_boundary)
    # split into hemispheres
    lat_masked = lat[mask]

    # 1. Find latitude of maximal (minimal) tropical Psi in the  (SH)
    Pmax_lat_masked = TropD_Calculate_MaxLat(P[..., subpolar_mask], lat[subpolar_mask])

    # 2. Find latitude of minimal (maximal) subtropical Psi in the  (SH)
    # poleward of tropical max (min)
    # define region poleward and Psi in region
    lat_after_Pmax = lat_masked >= Pmax_lat_masked[..., None]
    Plat_after_Pmax = np.where(lat_after_Pmax, P[..., mask], np.nan)

    Pmin_lat_masked = TropD_Calculate_MaxLat(-Plat_after_Pmax, lat_masked)

    # 3. Find the zero crossing between the above latitudes
    lat_in_between = (lat_masked <= Pmin_lat_masked[..., None]) & lat_after_Pmax
    Plat_in_between = np.where(lat_in_between, P[..., mask], np.nan)

    if method == "Psi_500_10Perc":
        Pmax = P[..., subpolar_mask].max(axis=-1)[..., None]
        Phi = TropD_Calculate_ZeroCrossing(Plat_in_between - 0.1 * Pmax, lat_masked)
    else:
        Phi = TropD_Calculate_ZeroCrossing(
            Plat_in_between, lat_masked, lat_uncertainty=lat_uncertainty
        )

    return dict(phi=Phi)


@hemisphere_handler
def TropD_Metric_PSL(
    ps: np.ndarray, 
    lat: np.ndarray, 
    method: str = "peak", 
    **maxlat_kwargs
) -> dict[str,np.ndarray]:
    """
    TropD Sea-level Pressure (PSL) Metric

    Latitude of maximum of the subtropical sea-level pressure

    Parameters
    ----------
    ps : np.ndarray (dim1, ..., lat)
        N-dimensional sea-level pressure
    lat : np.ndarray (lat,)
        latitude array
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
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if ``method="peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional latitudes of the subtropical PSL maximum
    """

    ps = np.asarray(ps)
    lat = np.asarray(lat)
    metric_value_bool = maxlat_kwargs.pop('metric_value', False)

    if ps.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of ps had shape {ps.shape[-1]},"
            f" not aligned with lat size {lat.size}"
        )
    if method not in ["max", "peak"]:
        raise ValueError("unrecognized method " + method)

    if method == "peak":
        maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
    else:  # max
        maxlat_kwargs["n"] = maxlat_kwargs.get("n", 6)
    # lat should already be last axis
    maxlat_kwargs.pop("axis", None)

    eq_boundary = 15
    polar_boundary = 60
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    Phi = TropD_Calculate_MaxLat(ps[..., mask], lat[mask], **maxlat_kwargs)

    if metric_value_bool:
        psmax = np.zeros(ps.shape[:-1])
        ps_flat = ps.reshape(-1, lat.size)
        i_lat_max = [np.abs(lat-Phi.flat[i]).argmin() for i in range(np.size(Phi))] 

        for i in range(len(i_lat_max)):
            phi_ind = np.unravel_index(i, ps.shape[:-1])
            psmax[phi_ind] = ps_flat[i,i_lat_max[i]]
        return dict(phi=Phi, metric_value=psmax)
    else:
        return dict(phi=Phi)


@hemisphere_handler
def TropD_Metric_STJ(
    U: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "adjusted_peak",
    n_fit: int = 1,
    **max_kwargs,
) -> dict[str,np.ndarray]:
    #TODO: Amend fit to adjusted_fit for consistency. Deprecate and add warning
    """
    TropD Subtropical Jet (STJ) Metric

    Latitude of the (adjusted) maximum upper-level zonal-mean zonal wind

    Parameters
    ----------
    U : numpy.ndarray (dim1,...,lat,lev,)
        N-dimensional zonal-mean zonal wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical level array in hPa
    method : {"adjusted_peak", "adjusted_max", "core_peak", "core_max", "fit"}, optional
        Method for determing the latitude of the STJ maximum, by default "adjusted_peak":

        * "adjusted_peak": Latitude of maximum (smoothing parameter``n=30``) of [the
                           zonal wind averaged between 100 and 400 hPa] minus [the zonal
                           mean zonal wind at the level closest to 850hPa], poleward of
                           10 degrees and equatorward of the Eddy Driven Jet latitude
        * "adjusted_max": Latitude of maximum (smoothing parameter ``n=6``) of [the
                          zonal wind averaged between 100 and 400 hPa] minus [the zonal
                          mean zonal wind at the level closest to 850hPa], poleward of 10
                          degrees and equatorward of the Eddy Driven Jet latitude
        * "core_peak": Latitude of maximum (smoothing parameter ``n=30``) of the zonal
                       wind averaged between 100 and 400 hPa, poleward of 10 degrees and
                       equatorward of 60 degrees
        * "core_max": Latitude of maximum (smoothing parameter ``n=6``) of the zonal wind
                      averaged between 100 and 400 hPa, poleward of 10 degrees and
                      equatorward of 60 degrees
         * "fit": Latitude of the maximum of [the zonal wind averaged between 100 and 400
                  hPa] minus [the zonal mean zonal wind at the level closest to 850hPa]
                  using a quadratic polynomial fit of data from grid points surrounding
                  the grid point of the maximum

    n_fit : int, optional
        used when ``method="fit"``, determines the number of points around the max to use
        while fitting, by default 1

    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="fit"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="core_max"`` or
            ``method="adjusted_max"``, 30 if ``method="core_peak"`` or
            ``method="adjusted_peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the STJ
    Umax : numpy.ndarray (dim1, ..., dimN-2), conditional
        N-2 dimensional  STJ strength, returned if ``method="fit"``
    """

    U = np.asarray(U)
    if U.ndim < 3:
        U = U[None, ...]
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    #vertical_position_bool = max_kwargs.pop('vertical_position', False)
    metric_value_bool = max_kwargs.pop('metric_value', False)
    
    if U.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"last axes of U had shape {U.shape[-2:]},"
            f" not aligned with grid shape {lat.size, lev.size}"
        )

    if method not in ["adjusted_peak", "core_peak", "adjusted_max", "core_max", "fit"]:
        raise ValueError("unrecognized method " + method)

    layer_400_to_100 = (lev >= 100) & (lev <= 400)
    lev_int = lev[layer_400_to_100]

    # Pressure weighted vertical mean of U
    if lev_int.size > 1:
        u_int = np.trapz(U[..., layer_400_to_100], lev_int, axis=-1) / (
            lev_int[-1] - lev_int[0]
        )
    else:
        u_int = U[..., layer_400_to_100]

    eq_boundary = 10
    polar_boundary = 60
    lat_mask = (lat > eq_boundary) & (lat < polar_boundary)
    lat_masked = lat[lat_mask]
    lev_masked = lev[layer_400_to_100]
    output_dict = {}

    if ("adjusted" in method) or method == "fit":  # adjusted_peak, adjusted_max methods
        idx_850 = find_nearest(lev, 850)
        u = u_int - U[..., idx_850]
        ##If vertical location of STJ is requested
        #if vertical_position_bool:
        #  u_vert = U[..., layer_400_to_100]-U[..., idx_850][...,None]
        #  u_vert_masked = u_vert[...,lat_mask,:]
    else:  # core_peak, core_max methods
        u = u_int
        ##If vertical location of STJ is requested
        #if vertical_position_bool:
        #  u_vert = U[..., layer_400_to_100]
        #  u_vert_masked = u_vert[...,lat_mask,:]

    u_masked = u[..., lat_mask]

    if method != "fit":

        if "peak" in method:  # adjusted_peak or core_peak have different default
            max_kwargs["n"] = max_kwargs.get("n", 30)
        else:  # adjusted_max or core_max methods
            max_kwargs["n"] = max_kwargs.get("n", 6)
        # lat should already be last axis
        max_kwargs.pop("axis", None)
        
        if "adjusted" in method:  # adjusted_max or adjusted_peak
            edj_metric_dict = TropD_Metric_EDJ(U, lat, lev)
            #get the latitude metric from edj metric dict
            Phi_EDJ = edj_metric_dict[list(edj_metric_dict.keys())[0]]
            lat_before_EDJ = lat_masked < Phi_EDJ[..., None]
            u_masked = np.where(lat_before_EDJ, u_masked, np.nan)

        u_flat = u_masked.reshape(-1, lat_masked.size)
        Phi = TropD_Calculate_MaxLat(u_masked, lat_masked, **max_kwargs)
        
        i_lat_max = [np.abs(lat_masked-Phi.flat[i]).argmin() for i in range(np.size(Phi))] 
        
        #if vertical_position_bool:
        #    u_vert_flat = u_vert_masked.reshape(-1, lat_masked.size, lev_masked.size)
        #    Pres_locations = TropD_Calculate_MaxPres(u_vert_flat, lev_masked , **max_kwargs)
        #   
        #    Umax = np.zeros(u.shape[:-1])
        #    P = np.zeros(u.shape[:-1])
        #    # Pressure level of max wind in the vertical at the location of the STJ in latitude
        #    for i in range(len(i_lat_max)):
        #      phi_ind = np.unravel_index(i, u.shape[:-1])
        #      pres_max_wind = Pres_locations[i, i_lat_max[i]]
        #      P[phi_ind] = pres_max_wind 
        #      if metric_value_bool:
        #        i_pres_max = np.abs(lev_masked-pres_max_wind).argmin() 
        #        Umax[phi_ind] = u_vert_flat[i,i_lat_max[i], i_pres_max]
        #elif metric_value_bool:
        if metric_value_bool:
            Umax = np.zeros(u.shape[:-1])
            for i in range(len(i_lat_max)):
                phi_ind = np.unravel_index(i, u.shape[:-1])
                Umax[phi_ind] = u_flat[i,i_lat_max[i]]

    else:  # method == 'fit':
        u_flat = u_masked.reshape(-1, lat_masked.size)
        
        Phi = np.zeros(u.shape[:-1])
        Umax = np.zeros(u.shape[:-1])
        P = np.zeros(u.shape[:-1])
        
        #if vertical_position_bool:
        #    u_vert_flat = u_vert_masked.reshape(-1, lat_masked.size, lev_masked.size)

        for i, Uh in enumerate(u_flat):
            Im = np.nanargmax(Uh)
            phi_ind = np.unravel_index(i, u.shape[:-1])
            
            if Im == 0 or Im == Uh.size - 1:
                Phi[phi_ind] = lat_masked[Im]
                continue

            if n_fit > Im or n_fit > Uh.size - Im + 1:
                N = np.min(Im, Uh.size - Im + 1)
            else:
                N = n_fit

            p = polynomial.polyfit(
                lat_masked[Im - N : Im + N + 1], Uh[Im - N : Im + N + 1], deg=2
            )
            # vertex of quadratic ax**2+bx+c is at -b/2a
            # p[0] + p[1]*x + p[2]*x**2
            Phi[phi_ind] = -p[1] / (2.0 * p[2])
            # value at vertex is (4ac-b**2)/(4a)
            Umax[phi_ind] = (4.0 * p[2] * p[0] - p[1] * p[1]) / 4.0 / p[2]

            #if vertical_position_bool:
            #    i_lat_max = np.abs(lat_masked-Phi[phi_ind]).argmin()
            #    #vertical profile of the wind at the location of stj in lat
            #    Uv = u_vert_flat[i,i_lat_max,:].squeeze()

            #    Im_v = np.nanargmax(Uv)

            #    if Im_v == 0 or Im_v == Uv.size - 1:
            #        P[phi_ind] = lev_masked[Im_v]
            #        continue

            #    if n_fit > Im_v or n_fit > Uv.size - Im_v + 1:
            #        N_v = np.min(Im_v, Uv.size - Im_v + 1)
            #    else:
            #        N_v = n_fit
            #    p_v = polynomial.polyfit(
            #        lev_masked[Im_v - N_v : Im_v + N_v + 1], Uv[Im_v - N_v : Im_v + N_v + 1], deg=2
            #    )
            #    # vertex of quadratic ax**2+bx+c is at -b/2a
            #    # p_v[0] + p_V[1]*x + p_v[2]*x**2
            #    P[phi_ind] = -p_v[1] / (2.0 * p_v[2])
            #    if metric_value_bool:
            #        # value at vertex is (4ac-b**2)/(4a)
            #        Umax[phi_ind] = (4.0 * p_v[2] * p_v[0] - p_v[1] * p_v[1]) / 4.0 / p_v[2]
    
    output_dict['phi'] = Phi
    #if vertical_position_bool: output_dict['vertical_position'] = P
    if metric_value_bool:  output_dict['metric_value'] = Umax
    return output_dict
                

@hemisphere_handler
def TropD_Metric_STJV(
    U: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "default",
    n_fit: int = 1,
    **kwargs,
) -> dict[str,np.ndarray]:
    #TODO: Amend fit to adjusted_fit for consistency. Deprecate and add warning
    """
    TropD Subtropical Jet Vertical (STJV) Metric

    Latitude of the (adjusted) maximum upper-level zonal-mean zonal wind

    Parameters
    ----------
    U : numpy.ndarray (dim1,...,lat,lev,)
        N-dimensional zonal-mean zonal wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical level array in hPa
    method : {"adjusted_peak", "adjusted_max", "core_peak", "core_max", "fit"}, optional
        Method for determing the latitude of the STJV maximum, by default "adjusted_peak":

        * "adjusted": Latitude of maximum of [the
                          zonal wind averaged between 100 and 400 hPa] minus [the zonal
                           mean zonal wind at the level closest to 850hPa], poleward of
                           10 degrees and equatorward of the Eddy Driven Jet latitude
        * "default": Latitude of maximum of the zonal
                       wind averaged between 100 and 400 hPa, poleward of 10 degrees and
                       equatorward of 60 degrees

    **kwargs : optional
        additional keyword arguments 
        "metric_value": bool, default .False.. Determines if STJV strength is returned

        "interp_max": bool, default .False.. If True, linearly interpolate to find STJV strength. 
        Requires that metric_value=.True.


    Returns
    -------
    Dict:
    'phi' : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the STJ
    'level' : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional pressure level of the STJ
    'metric_value' : numpy.ndarray (dim1, ..., dimN-2), conditional
        N-2 dimensional  STJ strength
    """

    U = np.asarray(U) #(time, lev, lat)
    if U.ndim < 3:
        U = U[None, ...]
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    
    if U.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"last axes of U had shape {U.shape[-2:]},"
            f" not aligned with grid shape {lat.size, lev.size}"
        )
    
    metric_value_bool = kwargs.pop('metric_value', False)
    interp_max_bool = kwargs.pop('interp_max', False)
   
    #order latitudes from eq to pole
    if lat[0] > lat[1]:
        lat = lat[::-1]
        U = U[...,::-1,:]
    
    layer_400_to_100 = (lev >= 100) & (lev <= 400)
    
    eq_boundary = kwargs.pop("tropical_boundary", 10.0)
    polar_boundary = kwargs.pop("polar_boundary", 60.0)
    lat_mask = (lat > eq_boundary) & (lat < polar_boundary)
    lat_masked = lat[lat_mask]
    lev_masked = lev[layer_400_to_100]
    u_masked_lev = U[...,layer_400_to_100]
    u_masked = u_masked_lev[...,lat_mask,:]
    
    #find latitude of STJ
    # Pressure weighted vertical mean of U
    if lev_masked.size > 1:
        u_int = np.trapz(u_masked, lev_masked, axis=-1) / (
            lev_masked[-1] - lev_masked[0]
        )
    else:
        u_int = u_masked

    output_dict = {}
    
    if ("adjusted" in method):  # adjusted methods
        idx_850 = find_nearest(lev, 850)
        u_integrated = u_int - U[...,lat_mask,idx_850]
    else:  # default
        u_integrated = u_int

    u_int_masked = u_integrated

    if "adjusted" in method:  # adjusted
        edj_metric_dict = TropD_Metric_EDJ(U, lat, lev)
        #get the latitude metric from edj metric dict
        Phi_EDJ = edj_metric_dict[list(edj_metric_dict.keys())[0]]
        lat_before_EDJ = lat_masked < Phi_EDJ[..., None]
        u_int_masked = np.where(lat_before_EDJ, u_integrated, np.nan)

    u_int_flat = u_int_masked.reshape(-1, lat_masked.size)
    du_int_dlat = np.gradient(u_int_flat, axis=-1)
    
    Phi = TropD_Calculate_ZeroCrossing(du_int_dlat, lat_masked)
    
    output_dict['phi'] = Phi

    u_flat = u_masked.reshape(-1, *u_masked.shape[-2:])
    
    #interpolate zonal wind to the latitude of STJ
    u_new_flat = np.zeros([u_flat.shape[0],u_flat.shape[2]])
    for i in range(u_flat.shape[0]):
        u_new_flat[i,:] = interp1d(lat_masked, u_flat[i,:,:], axis=-2)(Phi[i])
    
    u_at_stjlat = u_new_flat.reshape([*u_masked.shape[:-2],len(lev_masked)])
    
    p_surface = 1013
    z = -np.log(lev_masked/p_surface)
    du_dlev = np.gradient(u_at_stjlat, z, axis=-1)
    
    lev_max = TropD_Calculate_ZeroCrossing(du_dlev, lev_masked)
     
    #interpolated zonal wind at latitude of STJ to level of STJ max
    if metric_value_bool:
        u_max_flat = np.zeros(u_flat.shape[0])
        for i in range(u_flat.shape[0]):
            if interp_max_bool:
                z_max = -np.log(lev_max[i]/p_surface)
                u_max_flat[i] = interp1d(z, u_new_flat[i,:], axis=-1)(z_max)
            else:
                #nearest grid point to lat max
                i_lat_max = np.abs(lat_masked-Phi[i]).argmin()
                i_lev_max = np.abs(lev_masked-lev_max[i]).argmin()
                u_max_flat[i] = u_flat[i,i_lat_max,i_lev_max].squeeze()

        Umax = u_max_flat.reshape([*u_masked.shape[:-2],])
        

    output_dict['level'] = lev_max
    if metric_value_bool:  output_dict['metric_value'] = Umax

    return output_dict



@hemisphere_handler
def TropD_Metric_TPB(
    T: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "max_gradient",
    Z: Optional[np.ndarray] = None,
    Cutoff: float = 1.5e4,
    **maxlat_kwargs,
) -> dict[str,np.ndarray]:
    """
    TropD Tropopause Break (TPB) Metric

    Finds the latitude of the tropopause break

    Parameters
    ----------
    T : numpy.ndarray (dim1, ..., lat, lev)
        N-dimensional temperature array (K)
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical levels array in hPa
    method : {"max_gradient", "cutoff", "max_potemp"}, optional
        Method to identify tropopause break, by default "max_gradient":

        * "max_gradient": The latitude of maximal poleward gradient of the tropopause
                          height

        * "cutoff": The most equatorward latitude where the tropopause crosses
                    a prescribed cutoff value

        * "max_potemp": The latitude of maximal difference between the potential
                        temperature at the tropopause and at the surface

    Z : Optional[numpy.ndarray], optional
        N-dimensional geopotential height array (m), required by ``method="cutoff"``
    Cutoff : float, optional
        Geopotential height cutoff (m) that marks the location of the tropopause break,
        by default 1.5e4, used when ``method="cutoff"``
    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="cutoff"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max_gradient"``, 30 if ``method="max_pottemp"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the tropopause break
    """

    T = np.asarray(T)
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if T.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"final dimensions on T {T.shape[-2:]} and grid "
            f"coordinates don't match ({lat.size},{lev.size})"
        )
    if method not in ["max_gradient", "max_potemp", "cutoff"]:
        raise ValueError("unrecognized method " + method)
    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        lat = lat[::-1]
        T = T[..., ::-1, :]
        if Z is not None:
            Z = Z[..., ::-1, :]

    polar_boundary = 60.0
    eq_boundary = 0.0
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    if "max_" in method:  # 'max_gradient' or 'max_potemp'
        if method == "max_potemp":
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
            PT = T / (lev / 1000.0) ** KAPPA
            Pt, PTt = TropD_Calculate_TropopauseHeight(T, lev, Z=PT)
            F = PTt - np.nanmin(PT, axis=-1)
        else:
            Pt = TropD_Calculate_TropopauseHeight(T, lev, Z=None)
            F = np.diff(Pt, axis=-1) / (lat[1] - lat[0])
            lat = (lat[1:] + lat[:-1]) / 2.0
            F *= np.sign(lat)
            # redefine mask b/c we have new grid points
            mask = (lat > eq_boundary) & (lat < polar_boundary)
        F = np.where(np.isfinite(F), F, 0.0)

        Phi = TropD_Calculate_MaxLat(F[..., mask], lat[mask], **maxlat_kwargs)

    else:  # method == 'cutoff'
        if Z is None:
            raise ValueError('Z must be provided when method = "cutoff"')
        Pt, Ht = TropD_Calculate_TropopauseHeight(T, lev, Z)

        Phi = TropD_Calculate_ZeroCrossing(Ht[..., mask] - Cutoff, lat[mask])

    return dict(phi=Phi)


@hemisphere_handler
def TropD_Metric_UAS(
    U: np.ndarray,
    lat: np.ndarray,
    lev: Optional[np.ndarray] = None,
    method: str = "zero_crossing",
    lat_uncertainty: float = 0.0,
) -> dict[str,np.ndarray]:
    """
    TropD Near-Surface Zonal Wind (UAS) Metric

    Parameters
    ----------
    U : numpy.ndarray (dim1, ..., lat[, lev])
        N-dimensional zonal mean zonal wind array. Also accepts surface wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray, optional (lev,)
        vertical level array in hPa, required if U has final dimension lev
    method : {"zero_crossing"}, optional
        Method for identifying the surface wind zero crossing, by default "zero_crossing":

        * "zero_crossing": the first subtropical latitude where near-surface zonal wind
                           changes from negative to positive

    lat_uncertainty : float, optional
        the minimal distance allowed between adjacent zero crossings in degrees, by
        default 0.0

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional latitudes of the first subtropical zero crossing of
        near-surface zonal wind
    """

    U = np.asarray(U)
    lat = np.asarray(lat)
    get_lev = lev is not None
    U_grid_shape = U.shape[-2:] if get_lev else U.shape[-1]
    input_grid_shape = (lat.size, lev.size) if get_lev else lat.size  # type: ignore
    if U_grid_shape != input_grid_shape:
        raise ValueError(
            f"last axes of U w/ shape {U_grid_shape},"
            " not aligned with input grid of shape " + str(input_grid_shape)
        )
    if method != "zero_crossing":
        raise ValueError("unrecognized method " + method)

    if get_lev:
        uas = U[..., find_nearest(lev, 850)]  # type: ignore
    else:
        uas = U

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        uas = uas[..., ::-1]
        lat = lat[::-1]

    # define latitudes of boundaries certain regions
    eq_boundary = 5
    subpolar_boundary = 30
    polar_boundary = 60
    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    # get subtropical surface wind minimum
    uas_min_lat = TropD_Calculate_MaxLat(-uas[..., subpolar_mask], lat[subpolar_mask])

    # need to look for zero crossing after subtropical minimum,
    lat_after_Umin = lat[mask] > uas_min_lat[..., None]
    ulat_after_Umin = np.where(lat_after_Umin, uas[..., mask], np.nan)
    Phi = TropD_Calculate_ZeroCrossing(
        ulat_after_Umin, lat[mask], lat_uncertainty=lat_uncertainty
    )

    return dict(phi=Phi)


@hemisphere_handler
def TropD_Metric_Strat_TAL(
    upwelling: np.ndarray, 
    lat: np.ndarray, 
    method: str = "default",
    **kwargs
) -> dict[str,np.ndarray]:
    return TropD_Metric_TAL(upwelling, lat, **kwargs)
  
@hemisphere_handler
def TropD_Metric_Strat_CL(
    U: np.ndarray, 
    lat: np.ndarray, 
    method: str = "extratropics",
    **kwargs
) -> dict[str,np.ndarray]:
    return TropD_Metric_CL(U, lat, method=method, **kwargs)

@hemisphere_handler
def TropD_Metric_Strat_GWL(
    tracer: np.ndarray, 
    lat: np.ndarray, 
    zonal_mean_tracer=False,
    method: str = "default",
    **kwargs
) -> dict[str,np.ndarray]:
    return TropD_Metric_GWL(tracer, lat, zonal_mean_tracer=zonal_mean_tracer,**kwargs)

@hemisphere_handler
def TropD_Metric_Strat_ONESIGMA(
    tracer: np.ndarray,
    lat: np.ndarray,
    zonal_mean_tracer=False,
    method: str = "default",
    **kwargs
) -> dict[str,np.ndarray]:
    return TropD_Metric_ONESIGMA(tracer, lat, zonal_mean_tracer=zonal_mean_tracer,**kwargs)
  


## ========================================
## Stratospheric metrics
## Written by Kasturi Shah - August 3 2020
## converted to python by Alison Ming 8 April 2021
## vectorized and refactored by Sam Smith 19 May 22
#
#
#@hemisphere_handler
#def Shah_2020_GWL(
#    tracer: np.ndarray, 
#    lat: np.ndarray, 
#    zonal_mean_tracer=False
#) -> dict[str,np.ndarray]:
#    """
#    Computes the gradient-weighted latitude (GWL) from tracer data
#    Reference: Shah et al., JGR-A, 2020
#    https://doi.org/10.1029/2020JD033081
#
#    Parameters
#    ==========
#    tracer : numpy.ndarray (..., lat [, lon])
#        N-dimensional array of tracer data for computing gradient. If
#        ``zonal_mean_tracer=False``, the last dimension should correspond to the
#        longitude axis, otherwise it should be the latitude axis
#    latitude : numpy.ndarray (lat,)
#        latitude array in degrees
#    zonal_mean_tracer : bool, optional
#        whether the input tracer data is zonally symmetric (True) or zonally-varying
#        (False) (i.e., has a longitude dimension), by default False
#
#    Returns
#    =======
#    tracer_steep_lat: numpy.ndarray
#        N-2(N-1 for ``zonal_mean_tracer=True``) dimenional array of GWL latitudes in the
#        SH
#    """
#
#    tracer = np.asarray(tracer)
#    lat = np.asarray(lat).flatten()
#    if not zonal_mean_tracer:
#        tracer = tracer.swapaxes(-2, -1)
#    if lat.size != tracer.shape[-1]:
#        raise ValueError(
#            "input array 'lat' should be aligned with "
#            f"{'' if zonal_mean_tracer else 'second to '}last axis of tracer"
#        )
#    if lat[0] > lat[1]:
#        lat = lat[::-1]
#        tracer = tracer[..., ::-1]
#
#    phi_lat = np.radians(lat)
#    phi_mid_lat = 0.5 * (phi_lat[1:] + phi_lat[:-1])
#
#    # calculating gradients
#    grad_weight_lat = np.diff(tracer, axis=-1) / np.diff(phi_lat) * np.cos(phi_mid_lat)
#
#    # array of gradient weighted latitudes (...[,nlon])
#    GWL_lat = (phi_mid_lat * grad_weight_lat).sum(axis=-1) / grad_weight_lat.sum(
#        axis=-1
#    )
#    # area equivalent GWL width (in degrees)
#    if zonal_mean_tracer:
#        tracer_steep_lat = np.degrees(GWL_lat)
#    else:
#        tracer_steep_lat = np.degrees(np.arcsin(np.nanmean(np.sin(GWL_lat), axis=-1)))
#
#    return dict(phi=tracer_steep_lat)
#
#
#@hemisphere_handler
#def Shah_2020_1sigma(
#    tracer: np.ndarray,
#    lat: np.ndarray,
#    zonal_mean_tracer=False,
#) -> dict[str,np.ndarray]:
#    """
#    Computes the one-sigma width from 3-D tracer data
#    Reference: Shah et al., JGR-A, 2020
#    https://doi.org/10.1029/2020JD033081
#
#    Parameters
#    ==========
#    tracer: numpy.ndarray (...,lat[, lon])
#        N-dimensional array of tracer data for computing 1:math:`\sigma`-width. If
#        ``zonal_mean_tracer=False``, the last dimension should correspond to the
#        longitude axis, otherwise it should be the latitude axis
#    latitude: numpy.ndarray (lat,)
#        array of latitudes in degrees (1-D). If not increasing, it will be sorted
#    zonal_mean_tracer : bool, optional
#        whether the input tracer data is zonally symmetric (True) or zonally-varying
#        (False) (i.e., has a longitude dimension), by default False
#
#    Returns
#    =======
#    tracer_sigma_lat: numpy.ndarray
#        N-2(N-1 for ``zonal_mean_tracer=True``) dimenional array of
#        1:math:`\sigma`-widths in the NH
#    """
#
#    tracer = np.atleast_2d(tracer)
#    lat = np.asarray(lat).flatten()
#    if not zonal_mean_tracer:
#        tracer = tracer.swapaxes(-2, -1)
#    if lat.size != tracer.shape[-1]:
#        raise ValueError(
#            "input array 'lat' should be aligned with second to last axis of tracer"
#        )
#    if lat[0] > lat[1]:
#        lat = lat[::-1]
#        tracer = tracer[..., ::-1]
#
#    phi_lat = np.broadcast_to(np.radians(lat), tracer.shape)
#
#    # finding ranges of 70 degs latitude with largest tracer values
#    nlats_70deg = round(70.0 / (lat[1] - lat[0])) + 1
#    tracer_70deg_totals = fftconvolve(
#        np.where(np.isfinite(tracer), tracer, 0.0),
#        np.ones((tracer.ndim - 1) * (1,) + (nlats_70deg,)),
#        "valid",
#        axes=-1,
#    )
#    max_70deg_starts = np.argmax(tracer_70deg_totals, axis=-1)[..., None]
#    bands_70deg = (np.arange(lat.size) >= max_70deg_starts) & (
#        np.arange(lat.size) < max_70deg_starts + nlats_70deg
#    )
#    tracer_banded = np.ma.masked_where(~bands_70deg, tracer)
#    # mean and std of 70deg bands
#    mean_70deg = tracer_banded.mean(axis=-1).filled(0.0)
#    std_70deg = tracer_banded.std(axis=-1, ddof=1).filled(0.0)
#    threshold = (mean_70deg - std_70deg)[..., None]
#    sigma_width_lat = (
#        np.ma.masked_where(
#            ~((tracer < threshold) & (phi_lat > phi_lat[..., :1])),
#            phi_lat,
#            copy=False,
#        )
#        .min(axis=-1)
#        .filled(np.nan)
#    )
#    # area equivalent latitude (in degrees)
#    if zonal_mean_tracer:
#        tracer_sigma_lat = np.degrees(sigma_width_lat)
#    else:
#        tracer_sigma_lat = np.degrees(
#            np.arcsin(np.nanmean(np.sin(sigma_width_lat), axis=-1))
#        )
#
#    return dict(phi=tracer_sigma_lat)
