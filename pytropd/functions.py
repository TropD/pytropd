# Written by Ori Adam Mar.21.2017
# Edited by Alison Ming Jul.4.2017
# update to Python3, vectorized, and patched bugs - sjs 2.3.22
from typing import Callable, List, Optional, Tuple, Union, overload
import warnings
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

EARTH_RADIUS = 6371220.0
GRAV = 9.80616
GAS_CONSTANT_DRY = 287.04
SPEC_HEAT_PRES_DRY = 1005.7
KAPPA = GAS_CONSTANT_DRY / SPEC_HEAT_PRES_DRY


def find_nearest(
    array: np.ndarray, value: float, axis: Optional[int] = None, skipna: bool = False
) -> np.ndarray:
    """
    Find the index of the item in the array nearest to the value

    Parameters
    ----------
    array : numpy.ndarray
        array to search
    value : float
        value to be found
    axis : int, optional
        the axis of the array to search. if not given, the array is flattened prior to
        being searched. If given, output will be (n-1)-dimensional, returning the
        position of the value within the specified axis
    skipna : bool, optional
        whether to skip over NaN in the array, by default False

    Returns
    -------
    numpy.ndarray
        index(es) nearest to value in array
    """

    array = np.asarray(array)
    if (value < array.min()) | (value > array.max()):
        warnings.warn(
            "searching outside the bounds of an array will return "
            "index of the closest boundary",
            category=RuntimeWarning,
            stacklevel=2,
        )
    if not skipna:
        if np.isnan(array).any():
            warnings.warn(
                "searching an array with NaNs will return index "
                "of first NaN, use skipna=True to ignore NaNs",
                category=RuntimeWarning,
                stacklevel=2,
            )
        argmin: Callable = np.argmin
    else:
        argmin = np.nanargmin

    nearest_inds: np.ndarray = argmin(np.abs(array - value), axis=axis)  # type: ignore

    return nearest_inds


# Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_MaxLat(
    F: np.ndarray, lat: np.ndarray, n: int = 6, axis: int = -1
) -> np.ndarray:
    """
    Find latitude of absolute maximum value for a given interval

    *Note*: assumes a smoothly varying function

    Parameters
    ----------
    F : numpy.ndarray (dim1, ..., dimN-1, lat)
        N-dimensional array w/ lat as specified axis (default last), data assumed
        contingous with invalid data only on ends. interior nans are untested and will
        prompt warning
    lat : numpy.ndarray
        latitude array
    n : int, optional
        rank of moment used to calculate the position of max value.
            n = 1,2,4,6,8,... , by default 6
    axis : int, optional
        axis corresponding to latitude, by default -1

    Returns
    -------
    numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional array of latitude(s) of max values of F
    """

    # type casting
    F = np.asarray(F).copy()
    lat = np.asarray(lat)
    n = int(n)
    axis = int(axis)

    # input checking
    if n < 1:
        raise ValueError("The smoothing parameter n must be >= 1")
    if not np.isfinite(F).any():
        raise ValueError(
            "input field F has only NaN/inf values," " max lat cannot be computed"
        )
    if not F.shape[axis] == lat.size:
        raise ValueError(
            f"input field lat axis of size {F.shape[axis]} not "
            f"aligned with lat coordinates of size {lat.size}"
        )

    # ensure lat axis is last
    axes_list = list(range(F.ndim))
    axes_list[axis], axes_list[-1] = axes_list[-1], axes_list[axis]
    F = F.transpose(axes_list)

    # map F to [0,1]
    F -= np.nanmin(F, axis=-1)[..., None]
    F /= np.nanmax(F, axis=-1)[..., None]

    # in order for this function to handle "jagged" arrays (such as time
    # -varying domains, e.g. domains poleward/equatorward of another
    # circulation feature), it will receive all data on the same grid,
    # but with masked data outside the region of interest.
    # However, simply filling with zeros produces extra boundary values
    # in the integration, so we need to correct for this
    if not np.isfinite(F).all():
        F_filled = np.where(np.isfinite(F), F, 0.0)
        # find edges of nan regions
        nanbounds = np.diff(np.isfinite(F), axis=-1)
        # ensure we only have one contiguous region
        extra_nans_check = (nanbounds.sum(axis=-1) > 2.0).any()
        interior_nans_check = (np.isfinite(F)[..., 0] & np.isfinite(F)[..., -1]).any()
        if extra_nans_check or interior_nans_check:
            warnings.warn(
                "detected NaN/inf data located between valid data, "
                "this may not be handled correctly",
                stacklevel=2,
                category=RuntimeWarning,
            )

        # now construct arrays which are zero everywhere except on the
        # boundaries of nan regions. We can integrate these to correct for the
        # extra data added by trapz at boundaries
        pad = np.zeros_like(F[..., 0])[..., None]
        rbounds = np.where(nanbounds & np.isfinite(F[..., 1:]), F[..., 1:], 0.0)
        lbounds = np.where(nanbounds & np.isfinite(F[..., :-1]), F[..., :-1], 0.0)
        bounds = np.concatenate([pad, rbounds], axis=-1) + np.concatenate(
            [lbounds, pad], axis=-1
        )
        # now integrate and remove the extra boundary values added by trapz
        nom = (
            np.trapz(F_filled**n * lat, lat, axis=-1)
            - np.trapz(bounds**n * lat, lat, axis=-1) / 2.0
        )
        denom = (
            np.trapz(F_filled**n, lat, axis=-1)
            - np.trapz(bounds**n, lat, axis=-1) / 2.0
        )
        # weighted integral to account for discrete grid
        Ymax = nom / denom

    # if the grid is normal, just go ahead and integrate
    else:
        # weighted integral to account for discrete grid
        Ymax = np.trapz((F**n) * lat, lat, axis=-1) / np.trapz(F**n, lat, axis=-1)

    return Ymax


def TropD_Calculate_Mon2Season(
    Fm: np.ndarray,
    season: List[int],
    m: Optional[int] = None,
    first_jan_idx: int = 0,
    axis: int = 0,
) -> np.ndarray:
    """
    Calculate unweighted seasonal means from monthly time series. Dec is averaged with
    Jan and Feb of the **same** year, not the following year

    Parameters
    ----------
    Fm : numpy.ndarray (time, dim2, ..., dimN)
        N-dimensional array, only utilizes full years-worth of data will be utilized
    season : list of int
        0-based list of month indices, e.g., [11,0,1] for DJF
    m : int, optional
        deprecated. use first_jan_idx instead.
    first_jan_idx : int, optional
        index of first occurence of Jan in Fm, by default 0
    axis : int, optional
        time axis to resample along, by default 0

    Returns
    -------
    numpy.ndarray (time, dim2, ..., dimN)
        the annual time series of the seasonal means
        (not weighted for unequal month lengths)
    """

    Fm = np.asarray(Fm)
    season = list(season)
    if not all([ssn in range(12) for ssn in season]):
        raise ValueError("season can only include indices from 0 to 11")
    if m is not None:
        warnings.warn(
            "use of parameter m is deprecated and will be removed in"
            " future versions, please use first_jan_idx instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        first_jan_idx = m

    # ensure time first axis
    axes_list = list(range(Fm.ndim))
    axes_list[axis], axes_list[0] = axes_list[0], axes_list[axis]
    Fm = Fm.transpose(axes_list)

    whole_years = (Fm.shape[0] - first_jan_idx) // 12
    last_jan_idx = whole_years * 12 + first_jan_idx
    Fm = Fm[first_jan_idx:last_jan_idx, ...]
    F = (Fm.reshape(whole_years, 12, *Fm.shape[1:])[:, season]).mean(axis=1)

    return F


def TropD_Calculate_StreamFunction(
    V: np.ndarray, lat: np.ndarray, lev: np.ndarray
) -> np.ndarray:
    """
    Calculates the meridional mass streamfunction by integrating meridional wind from top
    of the atmosphere to surface

    Parameters
    ----------
    V : numpy.ndarray (dim1, ..., dimN-2, lat, lev)
        N-dimensional array of zonal-mean meridional wind with final 2 axes corresponding to latitude and vertical level, respectively
    lat : numpy.ndarray
        equally-spaced latitude array
    lev : numpy.ndarray
        vertical level array in hPa

    Returns
    -------
    numpy.ndarray (same shape as V)
        the meridional mass overturning streamfunction (psi)
    """

    if V.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"final dimensions on V {V.shape[-2:]} and grid"
            f" coordinates don't match ({lat.size},{lev.size})"
        )

    # mask any subsurface or missing data
    V = np.where(np.isfinite(V), V, 0)
    cos_lat = np.cos(lat * np.pi / 180.0)[:, None]
    # cumtrapz: if F(x) = int f(x) dx, return F(x) - F(x[0])
    psi = (
        cumtrapz(V, lev * 100.0, axis=-1, initial=0.0)
        * (EARTH_RADIUS / GRAV * 2.0 * np.pi)
        * cos_lat
    )

    return psi


@overload
def TropD_Calculate_TropopauseHeight(
    T: np.ndarray, P: np.ndarray, Z: None
) -> np.ndarray:
    ...


@overload
def TropD_Calculate_TropopauseHeight(
    T: np.ndarray, P: np.ndarray, Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ...


def TropD_Calculate_TropopauseHeight(
    T: np.ndarray, P: np.ndarray, Z: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the tropopause height in isobaric coordinates

    Based on the method in Birner (2010), according to the WMO definition:
    first level where the lapse rate <= 2K/km AND where the lapse rate <= 2K/km
    for at least 2km above that level

    Parameters
    ----------
    T : numpy.ndarray (dim1, ..., dimN-1, lev)
        N-dimensional temperature array
    P : numpy.ndarray (lev,)
        pressure levels in hPa
    Z : numpy.ndarray (same shape as T), optional
        N-dimensional array to be interpolated at tropopause, such as
        geopotential height (m) to yield tropopause height

    Returns
    -------
    numpy.ndarray (dim1, ..., dimN-1)
        the tropopause pressure in hPa
    numpy.ndarray (dim1, ..., dimN-1), optional
        returned if Z is provided. Corresponds to Z evaluated at the tropopause. If Z is geopotential height, it is tropopause altitude (m)
    """

    COMPUTE_Z = Z is not None

    T = np.atleast_2d(T)
    if COMPUTE_Z:
        Z = np.atleast_2d(Z)  # type: ignore
    if T.shape[-1] != P.size:
        raise ValueError(
            f"last axis of temperature data, size {T.shape[-1]}, "
            f"is not aligned with pressure data, size {P.size}"
        )

    # make P monotonically decreasing
    if P[-1] > P[0]:
        P = P[::-1]
        T = T[..., ::-1]
        if COMPUTE_Z:
            Z = Z[..., ::-1]  # type: ignore

    Pk = (P * 100.0) ** KAPPA
    Pk_mid = (Pk[:-1] + Pk[1:]) / 2.0

    T_mid = (T[..., :-1] + T[..., 1:]) / 2.0
    Gamma = (
        np.diff(T, axis=-1)
        / np.diff(Pk)
        * Pk_mid
        / T_mid
        * GRAV
        / SPEC_HEAT_PRES_DRY
        * 1000.0
    )  # K / km

    PI = (np.linspace(1000.0, 1.0, 1000) * 100.0) ** KAPPA
    interpG = interp1d(Pk_mid, Gamma, kind="linear", axis=-1, fill_value="extrapolate")
    GI = interpG(PI)
    interpT = interp1d(Pk_mid, T_mid, kind="linear", axis=-1, fill_value="extrapolate")
    TI = interpT(PI)
    # points in upper troposphere where lapse rate is less then 2km
    trop_layer = (
        (GI <= 2.0) & (PI < (550.0 * 100.0) ** KAPPA) & (PI > (75.0 * 100.0) ** KAPPA)
    )
    Pt = np.full_like(T[..., 0], np.nan)
    # loop over each individual column
    for i in range(Pt.size):
        icol = np.unravel_index(i, Pt.shape)
        if trop_layer[icol].any():
            # get pressure levels of potential points in the column
            idx = trop_layer[icol].nonzero()[0]
            Pidx = PI[idx]
            # get pressure level of 2km above potential tropopause pts
            Pidx_2km = Pidx - (
                2000.0 * GRAV / SPEC_HEAT_PRES_DRY / TI[icol + (idx,)] * Pidx
            )

            for c in range(Pidx.size):
                # get the idx of 2km above each point
                idx2km = find_nearest(PI, Pidx_2km[c])

                # if the lapse rate from the first point to 2km above
                # (inclusive) is less than 2, we've found our tropopause
                if (GI[icol + (slice(idx[c], idx2km + 1),)] <= 2.0).all():
                    Pt[icol] = Pidx[c]
                    break

    Pt = Pt ** (1.0 / KAPPA) / 100.0

    if COMPUTE_Z:
        Ht = np.zeros_like(Pt)
        # need to loop over individual columns again
        for i in range(Ht.size):
            icol = np.unravel_index(i, Ht.shape)
            f = interp1d(P, Z[icol], axis=-1)  # type: ignore
            Ht[icol] = f(Pt[icol])

        return Pt, Ht
    else:
        return Pt


# Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_ZeroCrossing(
    F: np.ndarray, lat: np.ndarray, lat_uncertainty: float = 0.0, axis: int = -1
) -> np.ndarray:
    """
    Find the first (with increasing index) zero crossing of the function F

    Parameters
    ----------
    F : numpy.ndarray (dim1, ..., dimN-1, lat)
        N dimensional array to search
    lat : numpy.ndarray (lat,)
        latitude array
    lat_uncertainty : float, optional
        same unit as lat. The minimum distance allowed between adjacent zero crossings of
        identical sign change, by default 0.0. For example, for lat_uncertainty = 10, if
        the most equatorward zero crossing is from positive to negative, NaN is returned
        if an additional zero crossing from positive to negative is found within 10
        degrees of the first one.
    axis : int, optional
        axis corresponding to latitude, by default -1 (last)

    Returns
    -------
    np.ndarray
        latitude(s) of zero crossing by linear interpolation
    """

    F = np.atleast_2d(F)
    lat = np.asarray(lat)
    lat_uncertainty = np.abs(lat_uncertainty)
    if F.shape[axis] != lat.size:
        raise ValueError(
            f"input array F with lat axis of size {F.shape[axis]}"
            f" is not aligned with lat coords of size {lat.size}"
        )
    if lat.size < 4:
        raise ValueError(
            "requires at least 4 latitudes to find zero " f"crossing, got {lat.size}"
        )

    # ensure lat axis is last
    axes_list = list(range(F.ndim))
    axes_list[axis], axes_list[-1] = axes_list[-1], axes_list[axis]
    F = F.transpose(axes_list)

    # Find all sign changes
    D = np.diff(np.sign(F), axis=-1)

    # initialize for looping
    ZC = np.full(D.shape[:-1], np.nan)
    # find zero crossings looping over all latitude bands
    for i in range(ZC.size):
        iband = np.unravel_index(i, ZC.shape)
        Di = D[iband]
        Fi = F[iband]

        # Make sure a zero crossing exists
        if not (np.any(Fi > 0) and np.any(Fi < 0)):
            continue  # no sign changes

        # get indices of all sign changes
        a = Di.nonzero()[0]
        a = a[np.isfinite(Di[a])]
        # If more than one zero crossing of same sign exists
        # in proximity to the first zero crossing.
        if a.size > 2:
            if np.abs(lat[a[2]] - lat[a[0]]) < lat_uncertainty:
                continue

        # first sign change
        if np.size(a) !=0:
            a1 = a[0]

            # if there is an exact zero, use its latitude...
            if np.abs(Di[a1]) == 1:
                ZC[iband] = lat[a1 + 1]
            else:  # np.abs(D[a1]) == 2 (directly from + to - or - to +)
                ZC[iband] = (
                    -Fi[a1] * (lat[a1 + 1] - lat[a1]) / (Fi[a1 + 1] - Fi[a1]) + lat[a1]
                )
        else:
            ZC[iband]=np.NaN

    return ZC
