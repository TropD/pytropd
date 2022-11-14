import numpy as np
import xarray as xr
from pandas import date_range
import pytropd as pyt
from pathlib import Path

# Validate metrics
# Check calculations with precalculated values from file within roundoff error
# Psi500
# read meridional velocity V(time,lat,lev), latitude and level
root = Path(__file__).absolute().parent.parent
data_dir = root / "ValidationData"
metrics_dir = root / "ValidationMetrics"


def get_validated_metric(metric: str) -> xr.DataArray:
    """
    for the given metric, open the validation file and reshape and label to match the
    data format output by the xarray methods

    Parameters
    ----------
    metric : str
        the metric file name to open

    Returns
    -------
    xr.DataArray
        the validated metric data
    """
    validated_metric = xr.open_dataset(metrics_dir / f"{metric}.nc")
    validated_metric = validated_metric.to_array(dim="hemsph")
    if "DJF" in metric:
        start_date = "1978-12-01"
    elif "MAM" in metric:
        start_date = "1979-03-01"
    elif "JJA" in metric:
        start_date = "1979-06-01"
    elif "SON" in metric:
        start_date = "1979-09-01"
    else:
        start_date = "1979-01-01"
    return validated_metric.assign_coords(
        hemsph=[h[-2:].upper() for h in validated_metric.hemsph.values],
        time=date_range(
            start=start_date, periods=validated_metric.time.size, freq="12MS"
        ),
    )


v_data = xr.open_dataset(data_dir / "va.nc")
u_data = xr.open_dataset(data_dir / "ua.nc")
comb_data = xr.merge([v_data, u_data])
comb_data = comb_data.rename(ua="u", va="v")
# Define a time axis
comb_data["time"] = date_range(
    start="1979-01-01", periods=v_data.sizes["time"], freq="MS"
)
# yearly mean of data
v_annual = comb_data.resample(time="AS").mean()

psi_metrics = v_annual.pyt_metrics.xr_psi()

validated_psi_metrics = get_validated_metric("PSI_ANN")
if np.allclose(*xr.align(psi_metrics, validated_psi_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated PSI metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated PSI metrics are NOT equal!")


# Tropopause break
# read temperature T(time,lat,lev), latitude and level
T_data = xr.open_dataset(data_dir / "ta.nc")
# Define a time axis
T_data["time"] = date_range(start="1979-01-01", periods=T_data.sizes["time"], freq="MS")
# yearly mean of data
T_annual = T_data.resample(time="AS").mean()

tpb_metrics_nh = T_annual.sortby("lat").sel(lat=slice(0, None)).pyt_metrics.xr_tpb()
tpb_metrics_sh = T_annual.sortby("lat").sel(lat=slice(None, 0)).pyt_metrics.xr_tpb()
tpb_metrics = xr.concat([tpb_metrics_nh, tpb_metrics_sh], "hemsph")

validated_tpb_metrics = get_validated_metric("TPB_ANN")
if np.allclose(*xr.align(tpb_metrics, validated_tpb_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated TPB metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated TPB metrics are NOT equal!")


# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
psl_data = xr.open_dataset(data_dir / "psl.nc")

psl_data["time"] = date_range(start="1979-01-01", periods=psl_data.time.size, freq="MS")
psl_seasonal = psl_data.resample(time="QS-DEC").mean()

psl_seasonal_metrics = psl_seasonal.pyt_metrics.xr_psl()

for ssn in ["DJF", "MAM", "JJA", "SON"]:
    validated_psl_metrics = get_validated_metric(f"PSL_{ssn}")
    psl_metrics = psl_seasonal_metrics.sel(time=psl_seasonal.time.dt.season == ssn)
    if ssn == "DJF":
        psl_metrics = psl_metrics.isel(time=slice(None, -1))
    if np.allclose(*xr.align(psl_metrics, validated_psl_metrics, join="outer")):
        print(f"OK. {ssn} Validation and calculated PSL metrics are the same!")
    else:
        if ssn != "DJF":
            print(
                f"Warning: {ssn} Validation and calculated PSL metrics are NOT equal!"
            )
        else:
            print(
                f"Warning: {ssn} Validation and calculated PSL metrics are NOT equal!"
                "\nHowever, this is expected because xarray computes seasonal averages "
                "for DJF differently than TropD_Calculate_Mon2Season"
            )


# Eddy driven jet
# read zonal wind U(time,lat,lev), latitude and level
u_data = xr.open_dataset(data_dir / "ua.nc")
# Define a time axis
u_data["time"] = date_range(start="1979-01-01", periods=u_data.sizes["time"], freq="MS")
# yearly mean of data
u_annual = u_data.resample(time="AS").mean()

edj_metrics = u_annual.sel(lev=850, method="nearest").pyt_metrics.xr_edj()

validated_edj_metrics = get_validated_metric("EDJ_ANN")
if np.allclose(*xr.align(edj_metrics, validated_edj_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated EDJ metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated EDJ metrics are NOT equal!")

# Subtropical jet
stj_metrics = u_annual.pyt_metrics.xr_stj()

validated_stj_metrics = get_validated_metric("STJ_ANN")
if np.allclose(*xr.align(stj_metrics, validated_stj_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated STJ metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated STJ metrics are NOT equal!")


# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
olr_data = -xr.open_dataset(data_dir / "rlnt.nc")
# Define a time axis
olr_data["time"] = date_range(
    start="1979-01-01", periods=olr_data.sizes["time"], freq="MS"
)
# yearly mean of data
olr_annual = olr_data.resample(time="AS").mean()

olr_metrics = olr_annual.pyt_metrics.xr_olr()

validated_olr_metrics = get_validated_metric("OLR_ANN")
if np.allclose(*xr.align(olr_metrics, validated_olr_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated OLR metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated OLR metrics are NOT equal!")


# P minus E
# read zonal mean monthly precipitation pr(time,lat)
pr_data = xr.open_dataset(data_dir / "pr.nc")
latent_heat_flux = xr.open_dataset(data_dir / "hfls.nc")
# use latent heat of vap to convert LHF to evap
er_data = -latent_heat_flux / 2510400.0
pe_data = (pr_data.pr - er_data.hfls).to_dataset(name="pe")
# Define a time axis
pe_data["time"] = date_range(
    start="1979-01-01", periods=pe_data.sizes["time"], freq="MS"
)
# yearly mean of data
pe_annual = pe_data.resample(time="AS").mean()

pe_metrics = pe_annual.pyt_metrics.xr_pe()
validated_pe_metrics = get_validated_metric("PE_ANN")
if np.allclose(*xr.align(pe_metrics, validated_pe_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated PE metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated PE metrics are NOT equal!")


# Surface winds
# read zonal mean surface wind U(time,lat)
uas_data = xr.open_dataset(data_dir / "uas.nc")
# Define a time axis
uas_data["time"] = date_range(
    start="1979-01-01", periods=uas_data.sizes["time"], freq="MS"
)
# yearly mean of data
uas_annual = uas_data.resample(time="AS").mean()

uas_metrics = uas_annual.pyt_metrics.xr_uas()

validated_uas_metrics = get_validated_metric("UAS_ANN")
if np.allclose(*xr.align(uas_metrics, validated_uas_metrics, join="outer")):
    print("OK. Annual-mean Validation and calculated UAS metrics are the same!")
else:
    print("Warning: annual-mean Validation and calculated UAS metrics are NOT equal!")
