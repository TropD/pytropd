import numpy as np
import xarray as xr
import pandas as pd
import pytropd as pyt
from pathlib import Path

# Validate metrics
# Check calculations with precalculated values from file within roundoff error
# Psi500
# read meridional velocity V(time,lat,lev), latitude and level
root = Path(__file__).absolute().parent.parent
data_dir = root / "ValidationData"
metrics_dir = root / "ValidationMetrics"

v_data = xr.open_dataset(data_dir / "va.nc")
u_data = xr.open_dataset(data_dir / "ua.nc")
comb_data = xr.merge([v_data, u_data])
comb_data = comb_data.rename(ua="u", va="v")

# Define a time axis
comb_data["time"] = pd.date_range(
    start="1979-01-01", periods=v_data.sizes["time"], freq="MS"
)

# yearly mean of data
v_annual = comb_data.groupby("time.year").mean("time")

psi_metrics = v_annual.pyt_metrics.xr_psi()

psi_metric_validation_data = xr.open_dataset(metrics_dir / "PSI_ANN.nc")

if (
    not (
        np.max(
            psi_metrics.psi.sel(metrics=0).values
            - psi_metric_validation_data.PSI_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            psi_metrics.psi.sel(metrics=1).values
            - psi_metric_validation_data.PSI_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated PSI metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated PSI metrics are the same!")


# Tropopause break
# read temperature T(time,lat,lev), latitude and level
T_data = xr.open_dataset(data_dir / "ta.nc")

# Define a time axis
T_data["time"] = pd.date_range(
    start="1979-01-01", periods=T_data.sizes["time"], freq="MS"
)

# yearly mean of data
T_annual = T_data.groupby("time.year").mean("time")

tpb_metrics = T_annual.pyt_metrics.xr_tpb()

tpb_metric_validation_data = xr.open_dataset(metrics_dir / "TPB_ANN.nc")

if (
    not (
        np.max(
            tpb_metrics.tpb.sel(metrics=0).values
            - tpb_metric_validation_data.TPB_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            tpb_metrics.tpb.sel(metrics=1).values
            - tpb_metric_validation_data.TPB_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated TPB metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated TPB metrics are the same!")


# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
psl_data = xr.open_dataset(data_dir / "psl.nc")

psl_data["time"] = pd.date_range(
    start="1979-01-01", periods=psl_data.sizes["time"], freq="MS"
)

psl_DJF_select = psl_data.sel(time=psl_data.time.dt.season == "DJF")
psl_MAM_select = psl_data.sel(time=psl_data.time.dt.season == "MAM")
psl_JJA_select = psl_data.sel(time=psl_data.time.dt.season == "JJA")
psl_SON_select = psl_data.sel(time=psl_data.time.dt.season == "SON")

# calculate mean per year
psl_DJF = psl_DJF_select.groupby(psl_DJF_select.time.dt.year).mean("time")
psl_MAM = psl_MAM_select.groupby(psl_MAM_select.time.dt.year).mean("time")
psl_JJA = psl_JJA_select.groupby(psl_JJA_select.time.dt.year).mean("time")
psl_SON = psl_SON_select.groupby(psl_SON_select.time.dt.year).mean("time")

psl_DJF_metrics = psl_DJF.pyt_metrics.xr_psl()
psl_MAM_metrics = psl_MAM.pyt_metrics.xr_psl()
psl_JJA_metrics = psl_JJA.pyt_metrics.xr_psl()
psl_SON_metrics = psl_SON.pyt_metrics.xr_psl()

psl_DJF_metric_validation_data = xr.open_dataset(metrics_dir / "PSL_DJF.nc")
psl_MAM_metric_validation_data = xr.open_dataset(metrics_dir / "PSL_MAM.nc")
psl_JJA_metric_validation_data = xr.open_dataset(metrics_dir / "PSL_JJA.nc")
psl_SON_metric_validation_data = xr.open_dataset(metrics_dir / "PSL_SON.nc")

if (
    not (
        np.max(
            psl_DJF_metrics.psl.sel(metrics=0).values
            - psl_DJF_metric_validation_data.PSL_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            psl_DJF_metrics.psl.sel(metrics=1).values
            - psl_DJF_metric_validation_data.PSL_NH.values
        )
    )
    < 1e-10
):
    print("Warning: DJF Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. DJF Validation and calculated PSL metrics are the same!")

if (
    not (
        np.max(
            psl_MAM_metrics.psl.sel(metrics=0).values
            - psl_MAM_metric_validation_data.PSL_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            psl_MAM_metrics.psl.sel(metrics=1).values
            - psl_MAM_metric_validation_data.PSL_NH.values
        )
    )
    < 1e-10
):
    print("Warning: MAM Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. MAM Validation and calculated PSL metrics are the same!")

if (
    not (
        np.max(
            psl_JJA_metrics.psl.sel(metrics=0).values
            - psl_JJA_metric_validation_data.PSL_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            psl_JJA_metrics.psl.sel(metrics=1).values
            - psl_JJA_metric_validation_data.PSL_NH.values
        )
    )
    < 1e-10
):
    print("Warning: JJA Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. JJA Validation and calculated PSL metrics are the same!")

if (
    not (
        np.max(
            psl_SON_metrics.psl.sel(metrics=0).values
            - psl_SON_metric_validation_data.PSL_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            psl_SON_metrics.psl.sel(metrics=1).values
            - psl_SON_metric_validation_data.PSL_NH.values
        )
    )
    < 1e-10
):
    print("Warning: SON Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. SON Validation and calculated PSL metrics are the same!")


# Eddy driven jet
# read zonal wind U(time,lat,lev), latitude and level
u_data = xr.open_dataset(data_dir / "ua.nc")

# Define a time axis
u_data["time"] = pd.date_range(
    start="1979-01-01", periods=u_data.sizes["time"], freq="MS"
)

# yearly mean of data
u_annual = u_data.groupby("time.year").mean("time")

edj_metrics = u_annual.pyt_metrics.xr_edj()

edj_metric_validation_data = xr.open_dataset(metrics_dir / "EDJ_ANN.nc")

if (
    not (
        np.max(
            edj_metrics.edj.sel(metrics=0).values
            - edj_metric_validation_data.EDJ_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            edj_metrics.edj.sel(metrics=1).values
            - edj_metric_validation_data.EDJ_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated EDJ metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated EDJ metrics are the same!")

# Subtropical jet
# yearly mean of data
u_annual = u_data.groupby("time.year").mean("time")

stj_metrics = u_annual.pyt_metrics.xr_stj()

stj_metric_validation_data = xr.open_dataset(metrics_dir / "STJ_ANN.nc")

if (
    not (
        np.max(
            stj_metrics.stj.sel(metrics=0).values
            - stj_metric_validation_data.STJ_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            stj_metrics.stj.sel(metrics=1).values
            - stj_metric_validation_data.STJ_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated STJ metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated STJ metrics are the same!")


# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
olr_data = xr.open_dataset(data_dir / "rlnt.nc")

# Define a time axis
olr_data["time"] = pd.date_range(
    start="1979-01-01", periods=olr_data.sizes["time"], freq="MS"
)

# yearly mean of data
olr_annual = olr_data.groupby("time.year").mean("time")

olr_metrics = olr_annual.pyt_metrics.xr_olr()

olr_metric_validation_data = xr.open_dataset(metrics_dir / "OLR_ANN.nc")

if (
    not (
        np.max(
            olr_metrics.olr.sel(metrics=0).values
            - olr_metric_validation_data.OLR_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            olr_metrics.olr.sel(metrics=1).values
            - olr_metric_validation_data.OLR_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated OLR metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated OLR metrics are the same!")

# P minus E
# read zonal mean monthly precipitation pr(time,lat)
pr_data = xr.open_dataset(data_dir / "pr.nc")
surf_latent_heat_flux_data = xr.open_dataset(data_dir / "hfls.nc")

# Latent heat of vaporization
L = 2510400.0

er_data = -surf_latent_heat_flux_data / L
pe_data = (pr_data.pr - er_data.hfls).to_dataset(name="pe")

# Define a time axis
pe_data["time"] = pd.date_range(
    start="1979-01-01", periods=pe_data.sizes["time"], freq="MS"
)

# yearly mean of data
pe_annual = pe_data.groupby("time.year").mean("time")

pe_metrics = pe_annual.pyt_metrics.xr_pe()

pe_metric_validation_data = xr.open_dataset(metrics_dir / "PE_ANN.nc")

if (
    not (
        np.max(
            pe_metrics.pe.sel(metrics=0).values - pe_metric_validation_data.PE_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            pe_metrics.pe.sel(metrics=1).values - pe_metric_validation_data.PE_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated PE metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated PE metrics are the same!")

# Surface winds
# read zonal mean surface wind U(time,lat)
uas_data = xr.open_dataset(data_dir / "uas.nc")

# Define a time axis
uas_data["time"] = pd.date_range(
    start="1979-01-01", periods=uas_data.sizes["time"], freq="MS"
)

# yearly mean of data
uas_annual = uas_data.groupby("time.year").mean("time")

uas_metrics = uas_annual.pyt_metrics.xr_uas()

uas_metric_validation_data = xr.open_dataset(metrics_dir / "UAS_ANN.nc")

if (
    not (
        np.max(
            uas_metrics.uas.sel(metrics=0).values
            - uas_metric_validation_data.UAS_SH.values
        )
    )
    < 1e-10
    and (
        np.max(
            uas_metrics.uas.sel(metrics=1).values
            - uas_metric_validation_data.UAS_NH.values
        )
    )
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated UAS metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated UAS metrics are the same!")
