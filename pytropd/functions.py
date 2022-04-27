# Written by Ori Adam Mar.21.2017
# Edited by Alison Ming Jul.4.2017
# update to Python3, vectorized, and patched bugs - sjs 2.3.22
import warnings
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

EARTH_RADIUS = 6371220.0
GRAV = 9.80616
GAS_CONSTANT_DRY = 287.04
SPEC_HEAT_PRES_DRY = 1005.7
KAPPA = GAS_CONSTANT_DRY / SPEC_HEAT_PRES_DRY


def find_nearest(array, value, axis=None, skipna=False):
    """
    Find the index of the item in the array nearest to the value

    Args:

        array: n-dimensional array

        value: value be found

    Kwargs:

        skipna: whether to skip over NaN in the array, default False

        axis: the axis of the array to search. if not given, the array is
              flattened prior to being searched. If given, output will be
              (n-1)-dimensional, returning the position of the value
              within the specified axis (optional)

    Returns:

        int: index of nearest value in array
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
        return np.argmin(np.abs(array - value), axis=axis)
    else:
        return np.nanargmin(np.abs(array - value), axis=axis)


# Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_MaxLat(F, lat, n=6, axis=-1):
    """
    Find latitude of absolute maximum value for a given interval
    ::note:: assumes smoothly varying function

    Args:

        F: N-dimensional array w/ lat as specified axis (default last), data
            assumed contingous with invalid data only on ends.
            interior nans are untested and will prompt warning

        lat: latitude array

        n (int): rank of moment used to calculate the position of max value.
                 n = 1,2,4,6,8,... (default 6)

        axis(int): axis corresponding to latitude, default last

    Returns:

        float: location of max value of F along lat
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
            np.trapz(F_filled ** n * lat, lat, axis=-1)
            - np.trapz(bounds ** n * lat, lat, axis=-1) / 2.0
        )
        denom = (
            np.trapz(F_filled ** n, lat, axis=-1)
            - np.trapz(bounds ** n, lat, axis=-1) / 2.0
        )
        # weighted integral to account for discrete grid
        Ymax = nom / denom

    # if the grid is normal, just go ahead and integrate
    else:
        # weighted integral to account for discrete grid
        Ymax = np.trapz((F ** n) * lat, lat, axis=-1) / np.trapz(F ** n, lat, axis=-1)

    return Ymax


def TropD_Calculate_Mon2Season(
    Fm, season, m=None, first_jan_idx=0, patch_indexing=True, axis=0
):
    """
    Calculate unweighted seasonal means from monthly time series

    Args:

        Fm: array of dimensions (time, dim2, dim3, ... dimN)
            currently only utilizes full years-worth of data

        season (list): 0-based list of month indices, e.g., [11,0,1] for DJF

        m (int): index of first occurence of Jan in Fm, assumed Fm begins
                 with Jan, deprecated, use first_jan_idx (optional)

        first_jan_idx (int): index of first occurence of Jan
                             in Fm, assumed Fm begins with Jan (optional)

        patch_indexing (bool): whether to use patch to fix buggy
                               indexing, default False to for consistency
                               w/ previous versions (optional)

    Returns:

        ndarray: the annual time series of the seasonal means
                 (not weighted for unequal month lengths), due to improper
                  indexing some input arrays may not behave as expected
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

    if patch_indexing:
        whole_years = (Fm.shape[0] - first_jan_idx) // 12
        last_jan_idx = whole_years * 12 + first_jan_idx
    # the original indexing method below results in improper handling
    # of data with length n*12 - 1, resulting in strange errors or bad
    # values (23 months only). errors also result when setting first_jan_idx
    else:
        last_jan_idx = (
            Fm.shape[0] - first_jan_idx + 1 - (Fm.shape[0] - first_jan_idx + 1) % 12
        )
    Fm = Fm[first_jan_idx:last_jan_idx, ...]
    F = 0
    for s in season:
        if patch_indexing:
            F += Fm[s::12, ...]
        else:
            F += Fm[first_jan_idx + s :: 12, ...]
    F /= len(season)

    return F


def TropD_Calculate_StreamFunction(V, lat, lev):
    """
    Calculate streamfunction by integrating meridional wind
    from top of the atmosphere to surface

    Args:

        V: array of zonal-mean meridional wind with dimensions
                (dim1, dim2, ..., dimN-2, lat, lev)

        lat: equally spaced latitude array

        lev: vertical level array in hPa

    Returns:

        ndarray: the streamfunction psi(dim1, dim2, ..., dimN-2,lat,lev)
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


def TropD_Calculate_TropopauseHeight(
    T, P, Z=None, use_local_idx=False, force_2km=False
):
    """
    Calculate the Tropopause Height in isobaric coordinates

    Based on the method in Birner (2010), according to the WMO definition:
    first level where the lapse rate <= 2K/km AND where the lapse rate <= 2K/km
    for at least 2km above that level

    Args:

        T(...,levels): N-dimensional temperature array with levels as last axis

        P: pressure levels in hPa, same dimension as last axis of T

        Z (optional): geopotential height (m) or another field
                      to be interpolated at tropopause, same dimensions as T

    Returns:

        Pt (ndarray): the tropopause level in hPa, with N-1 dimensions

        Optional (if Z is provided):

            Ht (ndarray): with N-1 dimensions. Ht is Z evaluated at the
                          tropopause. If Z is geopotential height (m), Ht is
                          the tropopause altitude (m)
    """

    COMPUTE_Z = Z is not None

    T = np.atleast_2d(T)
    if COMPUTE_Z:
        Z = np.atleast_2d(Z)
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
            Z = Z[..., ::-1]

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
            if use_local_idx:
                TIslicer = idx
            else:  # probable bug, current version uses surface temp
                TIslicer = slice(None, Pidx.size)
            Pidx_2km = Pidx - (
                2000.0 * GRAV / SPEC_HEAT_PRES_DRY / TI[icol + (TIslicer,)] * Pidx
            )

            for c in range(Pidx.size):
                # get the idx of 2km above each point
                # although find nearest returns the closest idx, which may
                # be less than 2km (in some cases much less)
                if force_2km:
                    # ensure that the layers we search are actually 2km
                    idx2km = find_nearest(PI, Pidx_2km[c])
                else:
                    idx2km = find_nearest(Pidx[c:], Pidx_2km[c]) + idx[c]

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
            f = interp1d(P, Z[icol], axis=-1)
            Ht[icol] = f(Pt[icol])

        return Pt, Ht
    else:
        return Pt


# Converted to python by Paul Staten Jul.29.2017
def TropD_Calculate_ZeroCrossing(
    F, lat, lat_uncertainty=0.0, axis=-1, patch_exact_zero=False
):
    """
    Find the first (with increasing index) zero crossing of the function F

    Args:

        F: N dimensional array with lat as final dim, or specified by axis

        lat: latitude array, same length as last axis of F

    Kwargs:

        lat_uncertainty (optional): (float) same unit as lat. The minimum
                                    distance allowed between adjacent zero
                                    crossings of identical sign change. For
                                    example, for lat_uncertainty = 10, if the
                                    most equatorward zero crossing is from
                                    positive to negative, NaN is returned if an
                                    additional zero crossing from positive to
                                    negative is found within 10 degrees of the
                                    first one.

        axis: latitude axis (default last)

        patch_exact_zero (optional): choose the exact latitude corresponding
                                     to 0 if True, otherwise return the grid
                                     point just prior (default, bug?)

    Returns:

        float: latitude of zero crossing by linear interpolation
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
        a1 = a[0]
        # if there is an exact zero, use its latitude...
        if np.abs(Di[a1]) == 1:
            ZC[iband] = lat[a1 + patch_exact_zero]
        else:  # np.abs(D[a1]) == 2 (directly from + to - or - to +)
            ZC[iband] = (
                -Fi[a1] * (lat[a1 + 1] - lat[a1]) / (Fi[a1 + 1] - Fi[a1]) + lat[a1]
            )

    return ZC
