# Written by Ori Adam Mar.21.2017
# Edited by Alison Ming Jul.4.2017
# rewrite for readability/vectorization - sjs 1.27.22
from typing import Optional, Tuple
import warnings
import numpy as np
from numpy.polynomial import polynomial
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from .functions import (
    find_nearest,
    TropD_Calculate_MaxLat,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_TropopauseHeight,
    TropD_Calculate_ZeroCrossing,
)

# kappa = R_dry / Cp_dry
KAPPA = 287.04 / 1005.7


def TropD_Metric_EDJ(
    U: np.ndarray,
    lat: np.ndarray,
    lev: Optional[np.ndarray] = None,
    method: str = "peak",
    n_fit: int = 1,
    **maxlat_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
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
    PhiSH : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional SH latitudes of the EDJ
    PHiNH : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional NH latitudes of the EDJ
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
    NHmask = (lat > eq_boundary) & (lat < polar_boundary)
    SHmask = (lat > -polar_boundary) & (lat < -eq_boundary)

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

        PhiNH = TropD_Calculate_MaxLat(u850[..., NHmask], lat[NHmask], **maxlat_kwargs)
        PhiSH = TropD_Calculate_MaxLat(u850[..., SHmask], lat[SHmask], **maxlat_kwargs)

    else:  # method == 'fit':
        u_flat = u850.reshape(-1, lat.size)

        Phi_list = []
        for hem_mask in [NHmask, SHmask]:
            lath = lat[hem_mask]
            Phi = np.zeros(u850.shape[:-1])
            for i, Uh in enumerate(u_flat[:, hem_mask]):
                Im = np.nanargmax(Uh)
                phi_ind = np.unravel_index(i, u850.shape[:-1])

                if Im == 0 or Im == Uh.size - 1:
                    Phi[phi_ind] = lath[Im]
                    continue

                if n_fit > Im or n_fit > Uh.size - Im + 1:
                    N = np.min(Im, Uh.size - Im + 1)
                else:
                    N = n_fit
                p = polynomial.polyfit(
                    lath[Im - N : Im + N + 1], Uh[Im - N : Im + N + 1], deg=2
                )
                # vertex of quadratic ax**2+bx+c is at -b/2a
                Phi[phi_ind] = -p[1] / (2 * p[2])
            Phi_list.append(Phi)

        PhiNH, PhiSH = Phi_list

    return PhiSH, PhiNH


def TropD_Metric_OLR(
    olr: np.ndarray,
    lat: np.ndarray,
    method: str = "250W",
    Cutoff: Optional[float] = None,
    **maxlat_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
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
    PhiSH : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional SH latitudes of the near-equator OLR max/threshold crossing
    PhiNH : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional NH latitudes of the near-equator OLR max/threshold crossing
    """

    olr = np.asarray(olr)
    lat = np.asarray(lat)
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

    NH_subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    SH_subpolar_mask = (lat > -subpolar_boundary) & (lat < -eq_boundary)

    if method == "peak":
        maxlat_kwargs["n"] = 30
    elif method == "max":
        maxlat_kwargs["n"] = 6
    # lat should already be last axis
    maxlat_kwargs.pop("axis", None)

    olr_max_lat_NH = TropD_Calculate_MaxLat(
        olr[..., NH_subpolar_mask], lat[NH_subpolar_mask], **maxlat_kwargs
    )
    olr_max_lat_SH = TropD_Calculate_MaxLat(
        olr[..., SH_subpolar_mask], lat[SH_subpolar_mask], **maxlat_kwargs
    )

    if method in ["cutoff", "250W", "20W", "10Perc"]:
        # get tropical OLR max for methods 20W and 10Perc
        olr_max_NH = olr[..., NH_subpolar_mask].max(axis=-1)[..., None]
        olr_max_SH = olr[..., SH_subpolar_mask].max(axis=-1)[..., None]

        # set cutoff dependent on method
        if method == "250W":
            NHCutoff = 250.0
            SHCutoff = 250.0
        elif method == "20W":
            NHCutoff = olr_max_NH - 20.0
            SHCutoff = olr_max_SH - 20.0
        elif method == "10Perc":
            NHCutoff = 0.9 * olr_max_NH
            SHCutoff = 0.9 * olr_max_SH
        else:  # method == cutoff
            NHCutoff = Cutoff  # type: ignore
            SHCutoff = Cutoff  # type: ignore

        # identify regions poleward of the OLR max in both hemispheres
        NHmask = (lat > eq_boundary) & (lat < polar_boundary)
        SHmask = (lat > -polar_boundary) & (lat < -eq_boundary)
        NH_max_mask = lat[NHmask] > olr_max_lat_NH[..., None]
        SH_max_mask = lat[SHmask] < olr_max_lat_SH[..., None]

        # OLR in each hemisphere, only valid poleward of max
        olr_NH = np.where(NH_max_mask, olr[..., NHmask], np.nan)
        olr_SH = np.where(SH_max_mask, olr[..., SHmask], np.nan)

        # get latitude where OLR falls below cutoff poleward of tropical max
        PhiNH = TropD_Calculate_ZeroCrossing(olr_NH - NHCutoff, lat[NHmask])
        PhiSH = TropD_Calculate_ZeroCrossing(
            olr_SH[..., ::-1] - SHCutoff, lat[SHmask][::-1]
        )
    # these methods don't need to find a threshold after
    # the OLR max, just need the max
    elif method in ["max", "peak"]:
        PhiNH = olr_max_lat_NH
        PhiSH = olr_max_lat_SH
    else:
        raise ValueError("unrecognized method " + method)

    return PhiSH, PhiNH


def TropD_Metric_PE(
    pe: np.ndarray,
    lat: np.ndarray,
    method: str = "zero_crossing",
    lat_uncertainty: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
    PhiSH : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional SH latitudes of the 1st subtropical P-E zero crossing
    PhiNH : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional NH latitudes of the 1st subtropical P-E zero crossing
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
    NHmask = (lat > eq_boundary) & (lat < polar_boundary)
    SHmask = (lat < -eq_boundary) & (lat > -polar_boundary)
    latNH = lat[NHmask]
    latSH = lat[SHmask]

    # find E-P maximum (P-E min) latitude in subtropics
    # first define the subpolar region to search, excluding poles due to low P-E
    NH_subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    SH_subpolar_mask = (lat < -eq_boundary) & (lat > -subpolar_boundary)
    Emax_latNH = TropD_Calculate_MaxLat(
        -pe[..., NH_subpolar_mask], lat[NH_subpolar_mask], n=30
    )
    Emax_latSH = TropD_Calculate_MaxLat(
        -pe[..., SH_subpolar_mask], lat[SH_subpolar_mask], n=30
    )

    # find zero crossings poleward of E-P max
    # flipping SH arrays to get the most equatorward zero crossing
    # first define regions poleward of E-P max in each hemisphere
    NH_after_Emax = latNH > Emax_latNH[..., None]
    SH_after_Emax = latSH < Emax_latSH[..., None]
    peNH_after_Emax = np.where(NH_after_Emax, pe[..., NHmask], np.nan)
    peSH_after_Emax = np.where(SH_after_Emax, pe[..., SHmask], np.nan)
    ZC1_latNH = TropD_Calculate_ZeroCrossing(
        peNH_after_Emax, latNH, lat_uncertainty=lat_uncertainty
    )
    ZC1_latSH = TropD_Calculate_ZeroCrossing(
        peSH_after_Emax[..., ::-1], latSH[::-1], lat_uncertainty=lat_uncertainty
    )

    # we've got the zero crossing poleward of E-P max, but it might not be the
    # right one. Now we need to go through and check the P-E gradient
    # to make sure P-E is increasing poleward.
    # if it is, use that latitude, else, use the next zero crossing

    # first check if the (northward) gradient value at the
    # zero crossing is increasing poleward
    peNH_increases_at_ZC = np.zeros_like(ZC1_latNH)
    peSH_increases_at_ZC = np.zeros_like(ZC1_latSH)
    pe_grad_flat = pe_grad.reshape(-1, lat.size)
    for i, ipe_grad in enumerate(pe_grad_flat):
        i_unrav = np.unravel_index(i, pe_grad.shape[:-1])
        interp_pe_grad = interp1d(lat, ipe_grad, axis=-1)
        peNH_increases_at_ZC[i_unrav] = interp_pe_grad(ZC1_latNH.flatten()[i]) > 0
        peSH_increases_at_ZC[i_unrav] = interp_pe_grad(ZC1_latSH.flatten()[i]) < 0

    # then get the next zero crossing for when we need it
    # first define regions poleward of zero crossing
    NH_after_ZC = latNH > ZC1_latNH[..., None]
    SH_after_ZC = latSH < ZC1_latSH[..., None]
    peNH_after_ZC = np.where(NH_after_ZC, pe[..., NHmask], np.nan)
    peSH_after_ZC = np.where(SH_after_ZC, pe[..., SHmask], np.nan)
    ZC2_latNH = TropD_Calculate_ZeroCrossing(
        peNH_after_ZC, latNH, lat_uncertainty=lat_uncertainty
    )
    ZC2_latSH = TropD_Calculate_ZeroCrossing(
        peSH_after_ZC, latSH, lat_uncertainty=lat_uncertainty
    )

    # if the gradient is increasing poleward, use it, otherwise, use the next
    PhiNH = np.where(peNH_increases_at_ZC, ZC1_latNH, ZC2_latNH)
    PhiSH = np.where(peSH_increases_at_ZC, ZC1_latSH, ZC2_latSH)

    return PhiSH, PhiNH


def TropD_Metric_PSI(
    V: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "Psi_500",
    lat_uncertainty: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
    PhiSH : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional SH latitudes of the Psi zero crossing
    PhiNH : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional NH latitudes of the Psi zero crossing
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
    NHmask = (lat > 0) & (lat < polar_boundary)
    SHmask = (lat < 0) & (lat > -polar_boundary)
    NH_subpolar_mask = (lat > 0) & (lat < subpolar_boundary)
    SH_subpolar_mask = (lat < 0) & (lat > -subpolar_boundary)
    # split into hemispheres
    latNH = lat[NHmask]
    latSH = lat[SHmask]

    # 1. Find latitude of maximal (minimal) tropical Psi in the NH (SH)
    Pmax_latNH = TropD_Calculate_MaxLat(P[..., NH_subpolar_mask], lat[NH_subpolar_mask])
    Pmax_latSH = TropD_Calculate_MaxLat(
        -P[..., SH_subpolar_mask], lat[SH_subpolar_mask]
    )

    # 2. Find latitude of minimal (maximal) subtropical Psi in the NH (SH)
    # poleward of tropical max (min)
    # define region poleward and Psi in region
    NH_after_Pmax = latNH >= Pmax_latNH[..., None]
    SH_after_Pmax = latSH <= Pmax_latSH[..., None]
    PNH_after_Pmax = np.where(NH_after_Pmax, P[..., NHmask], np.nan)
    PSH_after_Pmax = np.where(SH_after_Pmax, P[..., SHmask], np.nan)

    Pmin_latNH = TropD_Calculate_MaxLat(-PNH_after_Pmax, latNH)
    Pmin_latSH = TropD_Calculate_MaxLat(PSH_after_Pmax, latSH)

    # 3. Find the zero crossing between the above latitudes
    NH_in_between = (latNH <= Pmin_latNH[..., None]) & NH_after_Pmax
    SH_in_between = (latSH >= Pmin_latSH[..., None]) & SH_after_Pmax
    PNH_in_between = np.where(NH_in_between, P[..., NHmask], np.nan)
    PSH_in_between = np.where(SH_in_between, P[..., SHmask], np.nan)

    if method == "Psi_500_10Perc":
        PmaxNH = P[..., NH_subpolar_mask].max(axis=-1)[..., None]
        PminSH = P[..., SH_subpolar_mask].min(axis=-1)[..., None]
        PhiNH = TropD_Calculate_ZeroCrossing(PNH_in_between - 0.1 * PmaxNH, latNH)
        PhiSH = TropD_Calculate_ZeroCrossing(
            PSH_in_between[..., ::-1] - 0.1 * PminSH, latSH[::-1]
        )
    else:
        PhiNH = TropD_Calculate_ZeroCrossing(
            PNH_in_between, latNH, lat_uncertainty=lat_uncertainty
        )
        PhiSH = TropD_Calculate_ZeroCrossing(
            PSH_in_between[..., ::-1], latSH[::-1], lat_uncertainty=lat_uncertainty
        )

    return PhiSH, PhiNH


def TropD_Metric_PSL(
    ps: np.ndarray, lat: np.ndarray, method: str = "peak", **maxlat_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
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
    PhiSH : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional SH latitudes of the subtropical PSL maximum
    PhiNH : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional NH latitudes of the subtropical PSL maximum
    """

    ps = np.asarray(ps)
    lat = np.asarray(lat)
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
    NHmask = (lat > eq_boundary) & (lat < polar_boundary)
    SHmask = (lat > -polar_boundary) & (lat < -eq_boundary)

    PhiNH = TropD_Calculate_MaxLat(ps[..., NHmask], lat[NHmask], **maxlat_kwargs)
    PhiSH = TropD_Calculate_MaxLat(ps[..., SHmask], lat[SHmask], **maxlat_kwargs)

    return PhiSH, PhiNH


def TropD_Metric_STJ(
    U: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "adjusted_peak",
    **maxlat_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
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
    method : {"adjusted_peak", "adjusted_max", "core_peak", "core_max"}, optional
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
                       equatorward of 70 degrees
        * "core_max": Latitude of maximum (smoothing parameter ``n=6``) of the zonal wind
                      averaged between 100 and 400 hPa, poleward of 10 degrees and
                      equatorward of 70 degrees

    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat`

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="core_max"`` or
            ``method="adjusted_max"``, 30 if ``method="core_peak"`` or
            ``method="adjusted_peak"``

    Returns
    -------
    PhiSH : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional SH latitudes of the STJ
    PHiNH : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional NH latitudes of the STJ
    """

    U = np.asarray(U)
    if U.ndim < 3:
        U = U[None, ...]
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if U.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"last axes of U had shape {U.shape[-2:]},"
            f" not aligned with grid shape {lat.size, lev.size}"
        )

    if method not in ["adjusted_peak", "core_peak", "adjusted_max", "core_max"]:
        raise ValueError("unrecognized method " + method)

    layer_400_to_100 = (lev >= 100) & (lev <= 400)
    lev_int = lev[layer_400_to_100]

    # Pressure weighted vertical mean of U
    if lev_int.size > 1:
        u_int = np.trapz(U[..., layer_400_to_100], lev_int, axis=-1) / (
            lev_int[-1] - lev_int[0]
        )

    else:  # why take mean of 1 level?
        u_int = np.mean(U[..., layer_400_to_100], axis=-1)

    if "adjusted" in method:  # adjusted_peak, adjusted_max methods
        idx_850 = find_nearest(lev, 850)
        u = u_int - U[..., idx_850]

    else:  # core_peak, core_max methods
        u = u_int

    if "peak" in method:  # adjusted_peak or core_peak have different default
        maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
    else:  # adjusted_max or core_max methods
        maxlat_kwargs["n"] = maxlat_kwargs.get("n", 6)
    # lat should already be last axis
    maxlat_kwargs.pop("axis", None)

    eq_boundary = 10
    polar_boundary = 60
    NHmask = (lat > eq_boundary) & (lat < polar_boundary)
    SHmask = (lat > -polar_boundary) & (lat < -eq_boundary)
    uNH = u[..., NHmask]
    uSH = u[..., SHmask]
    if "adjusted" in method:  # adjusted_max or adjusted_peak
        PhiSH_EDJ, PhiNH_EDJ = TropD_Metric_EDJ(U, lat, lev)
        NH_before_EDJ = lat[NHmask] < PhiNH_EDJ[..., None]
        SH_before_EDJ = lat[SHmask] > PhiSH_EDJ[..., None]
        uNH = np.where(NH_before_EDJ, uNH, np.nan)
        uSH = np.where(SH_before_EDJ, uSH, np.nan)

    PhiNH = TropD_Calculate_MaxLat(uNH, lat[NHmask], **maxlat_kwargs)
    PhiSH = TropD_Calculate_MaxLat(uSH, lat[SHmask], **maxlat_kwargs)

    return PhiSH, PhiNH


def TropD_Metric_TPB(
    T: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "max_gradient",
    Z: Optional[np.ndarray] = None,
    Cutoff: float = 1.5e4,
    **maxlat_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    PhiSH : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional SH latitudes of the tropopause break
    PHiNH : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional NH latitudes of the tropopause break
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
    NHmask = (lat > eq_boundary) & (lat < polar_boundary)
    SHmask = (lat < -eq_boundary) & (lat > -polar_boundary)

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
            NHmask = (lat > eq_boundary) & (lat < polar_boundary)
            SHmask = (lat < -eq_boundary) & (lat > -polar_boundary)
        F = np.where(np.isfinite(F), F, 0.0)

        PhiNH = TropD_Calculate_MaxLat(F[..., NHmask], lat[NHmask], **maxlat_kwargs)
        PhiSH = TropD_Calculate_MaxLat(F[..., SHmask], lat[SHmask], **maxlat_kwargs)

    else:  # method == 'cutoff'
        if Z is None:
            raise ValueError('Z must be provided when method = "cutoff"')
        Pt, Ht = TropD_Calculate_TropopauseHeight(T, lev, Z)

        PhiNH = TropD_Calculate_ZeroCrossing(Ht[..., NHmask] - Cutoff, lat[NHmask])
        PhiSH = TropD_Calculate_ZeroCrossing(
            Ht[..., SHmask][..., ::-1] - Cutoff, lat[SHmask][::-1]
        )

    return PhiSH, PhiNH


def TropD_Metric_UAS(
    U: np.ndarray,
    lat: np.ndarray,
    lev: Optional[np.ndarray] = None,
    method: str = "zero_crossing",
    lat_uncertainty: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TropD Near-Surface Zonal Wind (UAS) Metric

    Parameters
    ----------
    U : np.ndarray (dim1, ..., lat[, lev])
        N-dimensional zonal mean zonal wind array. Also accepts surface wind
    lat : np.ndarray (lat,)
        latitude array
    lev : np.ndarray, optional (lev,)
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
    PhiSH : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional SH latitudes of the first subtropical zero crossing of
        near-surface zonal wind
    PHiNH : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional NH latitudes of the first subtropical zero crossing of
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
    NH_subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    SH_subpolar_mask = (lat < -eq_boundary) & (lat > -subpolar_boundary)
    NHmask = (lat > eq_boundary) & (lat < polar_boundary)
    SHmask = (lat < -eq_boundary) & (lat > -polar_boundary)

    # get subtropical surface wind minimum
    uas_min_lat_NH = TropD_Calculate_MaxLat(
        -uas[..., NH_subpolar_mask], lat[NH_subpolar_mask]
    )
    uas_min_lat_SH = TropD_Calculate_MaxLat(
        -uas[..., SH_subpolar_mask], lat[SH_subpolar_mask]
    )

    # need to look for zero crossing after subtropical minimum,
    # flipping SH arrays to find the most equatorward zero crossing
    NH_after_Umin = lat[NHmask] > uas_min_lat_NH[..., None]
    SH_after_Umin = lat[SHmask] < uas_min_lat_SH[..., None]
    uNH_after_Umin = np.where(NH_after_Umin, uas[..., NHmask], np.nan)
    uSH_after_Umin = np.where(SH_after_Umin, uas[..., SHmask], np.nan)
    PhiNH = TropD_Calculate_ZeroCrossing(
        uNH_after_Umin, lat[NHmask], lat_uncertainty=lat_uncertainty
    )
    PhiSH = TropD_Calculate_ZeroCrossing(
        uSH_after_Umin[..., ::-1], lat[SHmask][::-1], lat_uncertainty=lat_uncertainty
    )

    return PhiSH, PhiNH


# ========================================
# Stratospheric metrics
#

# Written by Kasturi Shah
# Last updated: August 3 2020
# converted to python by Alison Ming 8 April 2021


def Shah_et_al_2020_GWL_3D(
    tracer_3d_strat=None,
    lon=None,
    lat=None,
    pressure_strat=None,
    timepoints=None,
    *args,
    **kwargs,
):
    """Computes the gradient-weighted latitude (GWL) from 3-D tracer data
    Reference: Shah et al., JGR-A, 2020

    Parameters
    ==========
    tracer_3d_strat: numpy array (dimensions: lon x lat x pressure x time)
    longitude: numpy array in degrees (1-D)
    latitude: numpy array in degrees (1-D)
    pressure: numpy array (1-D)
    time: numpy array (1-D)

    Returns
    =======
    output: tuple
    * GWL width NH (dimensions: pressure x time)
    * GWL width SH (dimensions: pressure x time)

    Notes
    =====
    Note that this script assumes that the latitude array:
    (1) is IN DEGREES, and
    (2) starts with the SH & becomes increasingly positive, i.e. -90 to 90.
    """

    # dimensions of lon, lat, pressure, time
    nlon = len(lon)
    nlat = len(lat)
    npressure = len(pressure_strat)
    ntimepoints = len(timepoints)

    nlat = np.where(lat > 0)[0]
    slat = np.where(lat < 0)[0]
    tracer_3d_strat_nhem = tracer_3d_strat[:, nlat, :, :]
    tracer_3d_strat_shem = tracer_3d_strat[:, slat, :, :]

    lat90n = np.where(abs(lat[nlat] - 90) == np.min(abs(lat[nlat] - 90)))[0][0]
    lat90s = np.where(abs(lat[slat] + 90) == np.min(abs(lat[slat] + 90)))[0][0]

    # arrays for storing area-equivalent GWL widths
    tracer_steep_nhem_equiv = np.empty((npressure, ntimepoints))
    tracer_steep_nhem_equiv[:] = np.nan
    tracer_steep_shem_equiv = np.empty((npressure, ntimepoints))
    tracer_steep_shem_equiv[:] = np.nan

    lat_in_rad = np.deg2rad(lat)

    for dt in np.arange(ntimepoints):
        for p in np.arange(npressure):

            # arrays for storing the GWL widths
            gradient_weighted_lat_nhem = np.empty((nlon,))
            gradient_weighted_lat_nhem[:] = np.nan
            gradient_weighted_lat_shem = np.empty((nlon,))
            gradient_weighted_lat_shem[:] = np.nan

            for k in np.arange(nlon):

                # calculating gradients
                gradient_nhem = np.diff(
                    tracer_3d_strat_nhem[k, : lat90n + 1, p, dt].T
                ) / np.diff(lat_in_rad[nlat[: lat90n + 1]])
                gradient_shem = np.diff(
                    tracer_3d_strat_shem[k, lat90s:, p, dt].T
                ) / np.diff(lat_in_rad[slat[lat90s:]])
                gradient_weighted_lat_nhem[k] = np.sum(
                    lat_in_rad[nlat[:lat90n]]
                    * gradient_nhem
                    * np.cos(lat_in_rad[nlat[:lat90n]])
                ) / np.sum(gradient_nhem * np.cos(lat_in_rad[nlat[:lat90n]]))
                gradient_weighted_lat_shem[k] = np.sum(
                    lat_in_rad[slat[lat90s:-1]]
                    * gradient_shem
                    * np.cos(lat_in_rad[slat[lat90s:-1]])
                ) / np.sum(gradient_shem * np.cos(lat_in_rad[slat[lat90s:-1]]))
            # area equivalent latitude at this pressure and longitude
            # (in degrees)
            tracer_steep_nhem_equiv[p, dt] = np.rad2deg(
                np.arcsin(np.nansum(np.sin(gradient_weighted_lat_nhem)) / nlon)
            )
            tracer_steep_shem_equiv[p, dt] = np.rad2deg(
                np.arcsin(np.nansum(np.sin(gradient_weighted_lat_shem)) / nlon)
            )

    return tracer_steep_nhem_equiv, tracer_steep_shem_equiv


def Shah_et_al_2020_GWL_zonalmean(
    tracer_2d_strat=None,
    lat=None,
    pressure_strat=None,
    timepoints=None,
    *args,
    **kwargs,
):
    """Computes the gradient-weighted latitude (GWL) from zonal mean tracer data
    Reference: Shah et al., JGR-A, 2020

    Parameters
    ==========
    tracer_2d_strat: numpy array (dimensions: lat x pressure x time)
    latitude: numpy array in degrees (1-D)
    pressure: numpy array (1-D)
    time: numpy array (1-D)

    Returns
    =======
    output: tuple
    * GWL width NH (dimensions: pressure x time)
    * GWL width SH (dimensions: pressure x time)

    Notes
    =====
    Note that this script assumes that the latitude array:
    (1) is IN DEGREES, and
    (2) starts with the SH & becomes increasingly positive, i.e. -90 to 90.
    """

    # dimensions of lat, pressure, time
    nlat = len(lat)
    npressure = len(pressure_strat)
    ntimepoints = len(timepoints)

    nlat = np.where(lat > 0)[0]
    slat = np.where(lat < 0)[0]
    tracer_2d_strat_nhem = tracer_2d_strat[nlat, :, :]
    tracer_2d_strat_shem = tracer_2d_strat[slat, :, :]

    lat90n = np.where(abs(lat[nlat] - 90) == np.min(abs(lat[nlat] - 90)))[0][0]
    lat90s = np.where(abs(lat[slat] + 90) == np.min(abs(lat[slat] + 90)))[0][0]

    # arrays for storing area-equivalent GWL widths
    tracer_steep_nhem_equiv = np.empty((npressure, ntimepoints))
    tracer_steep_nhem_equiv[:] = np.nan
    tracer_steep_shem_equiv = np.empty((npressure, ntimepoints))
    tracer_steep_shem_equiv[:] = np.nan

    lat_in_rad = np.deg2rad(lat)

    for dt in np.arange(ntimepoints):
        for p in np.arange(npressure):
            # calculating gradients
            gradient_nhem = np.diff(
                tracer_2d_strat_nhem[: lat90n + 1, p, dt]
            ) / np.diff(lat_in_rad[nlat[: lat90n + 1]])
            gradient_shem = np.diff(tracer_2d_strat_shem[lat90s:, p, dt]) / np.diff(
                lat_in_rad[slat[lat90s:]]
            )

            tracer_steep_nhem_equiv[p, dt] = np.rad2deg(
                np.sum(
                    lat_in_rad[nlat[:lat90n]]
                    * gradient_nhem
                    * np.cos(lat_in_rad[nlat[:lat90n]])
                )
                / np.sum(gradient_nhem * np.cos(lat_in_rad[nlat[:lat90n]]))
            )

            tracer_steep_shem_equiv[p, dt] = np.rad2deg(
                np.sum(
                    lat_in_rad[slat[lat90s:-1]]
                    * gradient_shem
                    * np.cos(lat_in_rad[slat[lat90s:-1]])
                )
                / np.sum(gradient_shem * np.cos(lat_in_rad[slat[lat90s:-1]]))
            )

    return tracer_steep_nhem_equiv, tracer_steep_shem_equiv


def Shah_et_al_2020_one_sigma_3D(
    tracer_3d_strat=None,
    lon=None,
    lat=None,
    pressure_strat=None,
    timepoints=None,
    *args,
    **kwargs,
):
    """Computes the one-sigma width from 3-D tracer data
    Reference: Shah et al., JGR-A, 2020

    Parameters
    ==========
    tracer_3d_strat: numpy array (dimensions: lon x lat x pressure x time)
    longitude: numpy array in degrees (1-D)
    latitude: numpy array in degrees (1-D)
    pressure: numpy array (1-D)
    time: numpy array (1-D)

    Returns
    =======
    output: tuple
    * GWL width NH (dimensions: pressure x time)
    * GWL width SH (dimensions: pressure x time)

    Notes
    =====
    Note that this script assumes that the latitude array:
    (1) is IN DEGREES, and
    (2) starts with the SH & becomes increasingly positive, i.e. -90 to 90.
    """

    # dimensions of lon, lat, pressure, time
    nlon = len(lon)
    nlat = len(lat)
    npressure = len(pressure_strat)
    ntimepoints = len(timepoints)

    nlat = np.where(lat > 0)[0]
    slat = np.where(lat < 0)[0]
    tracer_3d_strat_nhem = tracer_3d_strat[:, nlat, :, :]
    tracer_3d_strat_shem = tracer_3d_strat[:, slat, :, :]

    # arrays for widths at each longitude
    tracer_sigma_nhem = np.empty((nlon, npressure, ntimepoints))
    tracer_sigma_nhem[:] = np.nan
    tracer_sigma_shem = np.empty((nlon, npressure, ntimepoints))
    tracer_sigma_shem[:] = np.nan

    tracer_sigma_nhem_equiv = np.empty((npressure, ntimepoints))
    tracer_sigma_nhem_equiv[:] = np.nan
    tracer_sigma_shem_equiv = np.empty((npressure, ntimepoints))
    tracer_sigma_shem_equiv[:] = np.nan

    lat_in_rad = np.deg2rad(lat)

    for dt in np.arange(ntimepoints):
        for pressure in np.arange(npressure):
            for k in np.arange(nlon):

                lat_delta = np.nanmean(np.diff(lat))
                gap_ind = int(round(70 / lat_delta))

                # finding range of 70degs with biggest max values
                maxval = np.empty(
                    (len(lat) - gap_ind),
                )
                maxval[:] = np.nan

                for ind in np.arange(len(lat) - gap_ind):
                    maxval[ind] = np.nansum(
                        tracer_3d_strat[k, ind : ind + gap_ind + 1, pressure, dt]
                    )

                a = np.nanmax(maxval)
                maxind = np.where(maxval == a)[0][0]

                # mean and std 35N-35S
                mean70deg = np.nanmean(
                    tracer_3d_strat[k, maxind : (maxind + gap_ind + 1), pressure, dt]
                )
                std70deg = np.nanstd(
                    tracer_3d_strat[k, maxind : (maxind + gap_ind + 1), pressure, dt],
                    ddof=1,
                )

                threshold = mean70deg - std70deg

                # finding latitudes less than this threshold
                nlatless = np.where(
                    tracer_3d_strat_nhem[k, :, pressure, dt] < threshold
                )[0]
                slatless = np.where(
                    tracer_3d_strat_shem[k, :, pressure, dt] < threshold
                )[0]

                if np.size(nlatless) != 0:
                    if nlatless[0] != 0:
                        tracer_sigma_nhem[k, pressure, dt] = lat_in_rad[
                            nlat[nlatless[0]]
                        ]
                    else:  # if lowest value is the equator, pick second one
                        if len(nlatless) > 1:
                            tracer_sigma_nhem[k, pressure, dt] = lat_in_rad[
                                nlat[nlatless[1]]
                            ]
                if np.size(slatless) != 0:
                    if slatless[-1] != slat[-1]:
                        tracer_sigma_shem[k, pressure, dt] = lat_in_rad[
                            slat[slatless[-1]]
                        ]
                    else:
                        if len(slatless) > 1:
                            tracer_sigma_shem[k, pressure, dt] = lat_in_rad[
                                slat[slatless[-2]]
                            ]

            # area equivalent latitude at this pressure and time (in degrees)
            tracer_sigma_nhem_equiv[pressure, dt] = np.rad2deg(
                np.arcsin(np.nansum(np.sin(tracer_sigma_nhem[:, pressure, dt])) / nlon)
            )
            tracer_sigma_shem_equiv[pressure, dt] = np.rad2deg(
                np.arcsin(np.nansum(np.sin(tracer_sigma_shem[:, pressure, dt])) / nlon)
            )

    return tracer_sigma_nhem_equiv, tracer_sigma_shem_equiv


def Shah_et_al_2020_one_sigma_zonalmean(
    tracer_2d_strat=None,
    lat=None,
    pressure_strat=None,
    timepoints=None,
    *args,
    **kwargs,
):
    """Computes the one-sigma width from zonal mean tracer data
    Reference: Shah et al., JGR-A, 2020

    Parameters
    ==========
    tracer_2d_strat: numpy array (dimensions: lat x pressure x time)
    latitude: numpy array in degrees (1-D)
    pressure: numpy array (1-D)
    time: numpy array (1-D)

    Returns
    =======
    output: tuple
    * GWL width NH (dimensions: pressure x time)
    * GWL width SH (dimensions: pressure x time)

    Notes
    =====
    Note that this script assumes that the latitude array:
    (1) is IN DEGREES, and
    (2) starts with the SH & becomes increasingly positive, i.e. -90 to 90.
    """

    # dimensions of lat, pressure, time
    npressure = len(pressure_strat)
    ntimepoints = len(timepoints)

    nlat = np.where(lat > 0)[0]
    slat = np.where(lat < 0)[0]
    tracer_2d_strat_nhem = tracer_2d_strat[nlat, :, :]
    tracer_2d_strat_shem = tracer_2d_strat[
        slat,
        :,
        :,
    ]

    # arrays for area-equivalent one sigma widths

    tracer_sigma_nhem_equiv = np.empty((npressure, ntimepoints))
    tracer_sigma_nhem_equiv[:] = np.nan
    tracer_sigma_shem_equiv = np.empty((npressure, ntimepoints))
    tracer_sigma_shem_equiv[:] = np.nan

    for dt in np.arange(timepoints):
        for pressure in np.arange(npressure):

            lat_delta = np.nanmean(np.diff(lat))
            gap_ind = int(round(70 / lat_delta))

            # finding range of 70degs with biggest max values
            maxval = np.empty(
                (len(lat) - gap_ind),
            )
            maxval[:] = np.nan
            for ind in np.arange(len(lat) - gap_ind):
                maxval[ind] = np.nansum(
                    tracer_2d_strat[ind : ind + gap_ind + 1, pressure, dt]
                )

            a = np.nanmax(maxval)
            maxind = np.where(maxval == a)[0][0]

            # mean and std 35N-35S
            mean70deg = np.nanmean(
                tracer_2d_strat[maxind : (maxind + gap_ind + 1), pressure, dt]
            )
            std70deg = np.nanstd(
                tracer_2d_strat[maxind : (maxind + gap_ind + 1), pressure, dt], ddof=1
            )

            threshold = mean70deg - std70deg

            # finding latitudes less than this threshold
            nlatless = np.where(tracer_2d_strat_nhem[:, pressure, dt] < threshold)[0]
            slatless = np.where(tracer_2d_strat_shem[:, pressure, dt] < threshold)[0]
            if np.size(nlatless) != 0:
                if nlatless[0] != 0:
                    tracer_sigma_nhem_equiv[pressure, dt] = lat[nlat[nlatless[0]]]
                else:  # if lowest value is the equator, pick second one
                    if len(nlatless) > 1:
                        tracer_sigma_nhem_equiv[pressure, dt] = lat[nlat[nlatless[1]]]
            if np.size(slatless) != 0:
                if slatless[-1] != slat[-1]:
                    tracer_sigma_shem_equiv[pressure, dt] = lat[slat[slatless[-1]]]
                else:
                    if len(slatless) > 1:
                        tracer_sigma_shem_equiv[pressure, dt] = lat[slat[slatless[-2]]]

    return tracer_sigma_nhem_equiv, tracer_sigma_shem_equiv
