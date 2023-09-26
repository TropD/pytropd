# Written by Alison Ming Jul.17.2023
# with contributions from Kasturi Shah and Molly Menzel
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

def TropD_Metric_TAL(
    upwelling: np.ndarray, 
    lat: np.ndarray, 
    method: str = "default",
    **kwargs
) -> dict[str,np.ndarray]:
    """
    Computes the location of the turnaround latitude from upwelling data

    Parameters
    ==========
    upwelling : numpy.ndarray (..., lat)
    latitude : numpy.ndarray (lat,)
        latitude array in degrees
    lat_uncertainty : float, optional
        the minimal distance allowed between adjacent zero crossings in degrees, by
        default 0.0

    Returns
    =======
    dict(phi=numpy.ndarray)
        dict of N-1 dimenional array of latitudes
    """

    upwelling = np.asarray(upwelling)
    lat = np.asarray(lat).flatten()
    lat_uncertainty = kwargs.pop("lat_uncertainty", 0)
    tropical_boundary = kwargs.pop("tropical_boundary", 10.0)
    if lat.size != upwelling.shape[-1]:
        raise ValueError(
            "input array 'lat' should be aligned with last axis of upwelling"
        )
    # order latitudes 
    if lat[0] > lat[1]:
        lat = lat[::-1]
        upwelling = upwelling[..., ::-1]


    mask = (lat > tropical_boundary) 
    Phi = TropD_Calculate_ZeroCrossing(upwelling[...,mask], lat[mask], lat_uncertainty=lat_uncertainty)
    return dict(phi=Phi)

def TropD_Metric_CL(
    U: np.ndarray, 
    lat: np.ndarray, 
    method: str = "extratropics",
    has_lev_axis: bool=False,
    **kwargs
) -> dict[str,np.ndarray]:
    """
    Computes the location of the critical latitude from zonal wind data

    Parameters
    ==========
    zonal wind : numpy.ndarray (..., lat,[lev: Optional])
    latitude : numpy.ndarray (lat,)
        latitude array in degrees
    

    method : {"extratropics", "second_crossing"}, optional
        * "extratropics": The latitude of first zero crossing poleward of the 
                          tropical boundary (default 10 degress) and polar boundary
                          of 89 degrees

        * "second_crossing": The most equatorward zero-crossing is first found. If
                          this lies within the tropical boundary (default 10 degress)
                          reject this and find the next one. Return this value even if
                          the second crossing is also within the tropical boundary
        * "all_zero_crossings": Return locations of all zero crossings if there is more than one. If the zonal wind is on different vertical levels, has_lev_axis needs to be set to True.  
    
    lat_uncertainty : float, optional
        the minimal distance allowed between adjacent zero crossings in degrees, by
        default 0.0

    Returns
    =======
    dict(phi=numpy.ndarray)
        dict of N-1 dimenional array of latitudes
    """
    U = np.asarray(U)
    lat = np.asarray(lat).flatten()
     

    lat_uncertainty = kwargs.pop("lat_uncertainty", 0)
    tropical_boundary = kwargs.pop("tropical_boundary", 10.0)
    polar_boundary = kwargs.pop("polar_boundary", 80.0)
    
    if lat.size != U.shape[-1]:
        raise ValueError(
            "input array 'lat' should be aligned with last axis of upwelling"
        )
    if method not in ["extratropics", "second_crossing","all_zero_crossings"]:
        raise ValueError("unrecognized method " + method)
    
    # order latitudes 
    if lat[0] > lat[1]:
        lat = lat[::-1]
        U = U[..., ::-1]

    # define latitudes of boundaries certain regions
    # find a zero crossing in the extratropics
    # by default between 10 deg and 89 deg
    if method == "extratropics":
        mask = (lat > tropical_boundary) & (lat < polar_boundary)
        Phi = TropD_Calculate_ZeroCrossing(U[...,mask], lat[mask], lat_uncertainty=lat_uncertainty)
    #method = second_crossing
    elif method == "second_crossing":
        # identify a zero crossing. If this is equatorward of the tropical boundary, find the next one.
        # accept this even if it is inside the tropical boundary
        mask = (lat > tropical_boundary) & (lat < polar_boundary)
        Phi = TropD_Calculate_ZeroCrossing(U[...,mask], lat[mask], lat_uncertainty=lat_uncertainty)

        if Phi < tropical_boundary:
            mask = (lat > Phi) & (lat < polar_boundary)
            Phi = TropD_Calculate_ZeroCrossing(U[...,mask], lat[mask], lat_uncertainty=lat_uncertainty)
    
    #method = all_crossings finds up to 3 zero_crossing locations
    else: 
        Phi = np.zeros(np.shape(U)[:-1])*np.NaN  
        if has_lev_axis:
            for i_lev in np.arange(np.shape(U)[-2]):
                most_equatorward_lat = 0.0
                count = 0
                while count<3:
                    if np.isnan(most_equatorward_lat): 
                        count=3
                    else:
                        mask = (lat > most_equatorward_lat) & (lat < polar_boundary)
                        if len(lat[mask])>=4:
                            most_equatorward_lat = TropD_Calculate_ZeroCrossing(U[0,...,i_lev,mask].squeeze(), lat[mask], lat_uncertainty=lat_uncertainty)[0]
                            Phi[count,...,i_lev] = most_equatorward_lat 
                        count+=1
        else:
            most_equatorward_lat = 0.0
            mask = (lat > most_equatorward_lat) & (lat < polar_boundary)
            most_equatorward_lat = TropD_Calculate_ZeroCrossing(U[...,mask], lat[mask], lat_uncertainty=lat_uncertainty)
            Phi = most_equatorward_lat 

    return dict(phi=Phi)

# ========================================
# Stratospheric metrics
# Written by Kasturi Shah - August 3 2020
# converted to python by Alison Ming 8 April 2021
# vectorized and refactored by Sam Smith 19 May 22


def TropD_Metric_GWL(
    tracer: np.ndarray, 
    lat: np.ndarray, 
    zonal_mean_tracer=False,
    method: str = "default",
    **kwargs
) -> dict[str,np.ndarray]:
    """
    Computes the gradient-weighted latitude (GWL) from tracer data
    Reference: Shah et al., JGR-A, 2020
    https://doi.org/10.1029/2020JD033081

    Parameters
    ==========
    tracer : numpy.ndarray (..., lat [, lon])
        N-dimensional array of tracer data for computing gradient. If
        ``zonal_mean_tracer=False``, the last dimension should correspond to the
        longitude axis, otherwise it should be the latitude axis
    latitude : numpy.ndarray (lat,)
        latitude array in degrees
    zonal_mean_tracer : bool, optional
        whether the input tracer data is zonally symmetric (True) or zonally-varying
        (False) (i.e., has a longitude dimension), by default False

    Returns
    =======
    tracer_steep_lat: numpy.ndarray
        N-2(N-1 for ``zonal_mean_tracer=True``) dimenional array of GWL latitudes in the
        SH
    """

    tracer = np.asarray(tracer)
    lat = np.asarray(lat).flatten()
    if not zonal_mean_tracer:
        tracer = tracer.swapaxes(-2, -1)
    if lat.size != tracer.shape[-1]:
        raise ValueError(
            "input array 'lat' should be aligned with "
            f"{'' if zonal_mean_tracer else 'second to '}last axis of tracer"
        )
    if lat[0] > lat[1]:
        lat = lat[::-1]
        tracer = tracer[..., ::-1]

    phi_lat = np.radians(lat)
    phi_mid_lat = 0.5 * (phi_lat[1:] + phi_lat[:-1])

    # calculating gradients
    grad_weight_lat = np.diff(tracer, axis=-1) / np.diff(phi_lat) * np.cos(phi_mid_lat)

    # array of gradient weighted latitudes (...[,nlon])
    GWL_lat = (phi_mid_lat * grad_weight_lat).sum(axis=-1) / grad_weight_lat.sum(
        axis=-1
    )
    # area equivalent GWL width (in degrees)
    if zonal_mean_tracer:
        tracer_steep_lat = np.degrees(GWL_lat)
    else:
        tracer_steep_lat = np.degrees(np.arcsin(np.nanmean(np.sin(GWL_lat), axis=-1)))

    return dict(phi=tracer_steep_lat)


def TropD_Metric_ONESIGMA(
    tracer: np.ndarray,
    lat: np.ndarray,
    zonal_mean_tracer=False,
    method: str = "default",
    **kwargs
) -> dict[str,np.ndarray]:
    """
    Computes the one-sigma width from 3-D tracer data
    Reference: Shah et al., JGR-A, 2020
    https://doi.org/10.1029/2020JD033081

    Parameters
    ==========
    tracer: numpy.ndarray (...,lat[, lon])
        N-dimensional array of tracer data for computing 1:math:`\sigma`-width. If
        ``zonal_mean_tracer=False``, the last dimension should correspond to the
        longitude axis, otherwise it should be the latitude axis
    latitude: numpy.ndarray (lat,)
        array of latitudes in degrees (1-D). If not increasing, it will be sorted
    zonal_mean_tracer : bool, optional
        whether the input tracer data is zonally symmetric (True) or zonally-varying
        (False) (i.e., has a longitude dimension), by default False

    Returns
    =======
    tracer_sigma_lat: numpy.ndarray
        N-2(N-1 for ``zonal_mean_tracer=True``) dimenional array of
        1:math:`\sigma`-widths in the NH
    """

    tracer = np.atleast_2d(tracer)
    lat = np.asarray(lat).flatten()
    if not zonal_mean_tracer:
        tracer = tracer.swapaxes(-2, -1)
    if lat.size != tracer.shape[-1]:
        raise ValueError(
            "input array 'lat' should be aligned with second to last axis of tracer"
        )
    if lat[0] > lat[1]:
        lat = lat[::-1]
        tracer = tracer[..., ::-1]

    phi_lat = np.broadcast_to(np.radians(lat), tracer.shape)

    # finding ranges of 70 degs latitude with largest tracer values
    nlats_70deg = round(70.0 / (lat[1] - lat[0])) + 1
    tracer_70deg_totals = fftconvolve(
        np.where(np.isfinite(tracer), tracer, 0.0),
        np.ones((tracer.ndim - 1) * (1,) + (nlats_70deg,)),
        "valid",
        axes=-1,
    )
    max_70deg_starts = np.argmax(tracer_70deg_totals, axis=-1)[..., None]
    bands_70deg = (np.arange(lat.size) >= max_70deg_starts) & (
        np.arange(lat.size) < max_70deg_starts + nlats_70deg
    )
    tracer_banded = np.ma.masked_where(~bands_70deg, tracer)
    # mean and std of 70deg bands
    mean_70deg = tracer_banded.mean(axis=-1).filled(0.0)
    std_70deg = tracer_banded.std(axis=-1, ddof=1).filled(0.0)
    threshold = (mean_70deg - std_70deg)[..., None]
    sigma_width_lat = (
        np.ma.masked_where(
            ~((tracer < threshold) & (phi_lat > phi_lat[..., :1])),
            phi_lat,
            copy=False,
        )
        .min(axis=-1)
        .filled(np.nan)
    )
    # area equivalent latitude (in degrees)
    if zonal_mean_tracer:
        tracer_sigma_lat = np.degrees(sigma_width_lat)
    else:
        tracer_sigma_lat = np.degrees(
            np.arcsin(np.nanmean(np.sin(sigma_width_lat), axis=-1))
        )

    return dict(phi=tracer_sigma_lat)
