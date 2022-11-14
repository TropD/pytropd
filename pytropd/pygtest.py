import numpy as np
import pygeode as pyg
import pytropd as pyt
from pathlib import Path

# Validate metrics
# Check calculations with precalculated values from file within roundoff error
# Psi500
# read meridional velocity V(time,lat,lev), latitude and level


# Define a time axis
time_axis = pyg.ModelTime365(
    values=np.arange(365 * 38), units="days", startdate=dict(year=1979, month=1, day=1)
)
# data is monthly mean
time_axis = time_axis.monthlymean().time

root = Path(__file__).absolute().parent.parent
data_dir = root / "ValidationData"
metrics_dir = root / "ValidationMetrics"
v_data = pyg.open(data_dir / "va.nc")
# make the axes into the right format
# the code expects a lat axis. v_data currently contains a named axis called lat
v_data = v_data.replace_axes(time=time_axis)

# yearly mean of data
v_annual = v_data.va.yearlymean()

psi_metrics = pyt.pyg_psi(v_annual)

psi_metric_validation_data = pyg.open(metrics_dir / "PSI_ANN.nc")

if (
    not (np.max(psi_metrics(metrics=0)[:] - psi_metric_validation_data.PSI_SH[:]))
    < 1e-10
    and (np.max(psi_metrics(metrics=1)[:] - psi_metric_validation_data.PSI_NH[:]))
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated PSI metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated PSI metrics are the same!")

# Tropopause break
# read temperature T(time,lat,lev), latitude and level
T_data = pyg.open(data_dir / "ta.nc")

T_data = T_data.replace_axes(time=time_axis)

# yearly mean of data
T_annual = T_data.ta.yearlymean()

tpb_metrics = pyt.pyg_tpb(T_annual)

tpb_metric_validation_data = pyg.open(metrics_dir / "TPB_ANN.nc")

if (
    not (np.max(tpb_metrics(metrics=0)[:] - tpb_metric_validation_data.TPB_SH[:]))
    < 1e-10
    and (np.max(tpb_metrics(metrics=1)[:] - tpb_metric_validation_data.TPB_NH[:]))
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated TPB metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated TPB metrics are the same!")


# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
psl_data = pyg.open(data_dir / "psl.nc")

# make the axes into the right format
psl_data = psl_data.replace_axes(time=time_axis)

# seasonal mean of data
psl_seasonal_mean = psl_data.psl.seasonalmean()

# calculate metrics
psl_DJF_metrics = pyt.pyg_psl(psl_seasonal_mean(season=1))
psl_MAM_metrics = pyt.pyg_psl(psl_seasonal_mean(season=2))
psl_JJA_metrics = pyt.pyg_psl(psl_seasonal_mean(season=3))
psl_SON_metrics = pyt.pyg_psl(psl_seasonal_mean(season=4))

psl_DJF_metric_validation_data = pyg.open(metrics_dir / "PSL_DJF.nc")
psl_MAM_metric_validation_data = pyg.open(metrics_dir / "PSL_MAM.nc")
psl_JJA_metric_validation_data = pyg.open(metrics_dir / "PSL_JJA.nc")
psl_SON_metric_validation_data = pyg.open(metrics_dir / "PSL_SON.nc")

if (
    not (
        np.max(psl_DJF_metrics(metrics=0)[:] - psl_DJF_metric_validation_data.PSL_SH[:])
    )
    < 1e-10
    and (
        np.max(psl_DJF_metrics(metrics=1)[:] - psl_DJF_metric_validation_data.PSL_NH[:])
    )
    < 1e-10
):
    print("Warning: DJF Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. DJF Validation and calculated PSL metrics are the same!")

if (
    not (
        np.max(psl_MAM_metrics(metrics=0)[:] - psl_MAM_metric_validation_data.PSL_SH[:])
    )
    < 1e-10
    and (
        np.max(psl_MAM_metrics(metrics=1)[:] - psl_MAM_metric_validation_data.PSL_NH[:])
    )
    < 1e-10
):
    print("Warning: MAM Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. MAM Validation and calculated PSL metrics are the same!")

if (
    not (
        np.max(psl_JJA_metrics(metrics=0)[:] - psl_JJA_metric_validation_data.PSL_SH[:])
    )
    < 1e-10
    and (
        np.max(psl_JJA_metrics(metrics=1)[:] - psl_JJA_metric_validation_data.PSL_NH[:])
    )
    < 1e-10
):
    print("Warning: JJA Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. JJA Validation and calculated PSL metrics are the same!")

if (
    not (
        np.max(psl_SON_metrics(metrics=0)[:] - psl_SON_metric_validation_data.PSL_SH[:])
    )
    < 1e-10
    and (
        np.max(psl_SON_metrics(metrics=1)[:] - psl_SON_metric_validation_data.PSL_NH[:])
    )
    < 1e-10
):
    print("Warning: SON Validation and calculated PSL metrics are NOT equal!")
else:
    print("OK. SON Validation and calculated PSL metrics are the same!")

# Eddy driven jet
# read zonal wind U(time,lat,lev), latitude and level
u_data = pyg.open(data_dir / "ua.nc")

# make the axes into the right format
u_data = u_data.replace_axes(time=time_axis, lat=pyg.Lat(u_data.lat[:]))

# yearly mean of data
u_annual = u_data.ua.yearlymean()

edj_metrics = pyt.pyg_edj(u_annual)

edj_metric_validation_data = pyg.open(metrics_dir / "EDJ_ANN.nc")

if (
    not (np.max(edj_metrics(metrics=0)[:] - edj_metric_validation_data.EDJ_SH[:]))
    < 1e-10
    and (np.max(edj_metrics(metrics=1)[:] - edj_metric_validation_data.EDJ_NH[:]))
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated EDJ metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated EDJ metrics are the same!")

# Subtropical jet

stj_metrics = pyt.pyg_stj(u_annual)

stj_metric_validation_data = pyg.open(metrics_dir / "STJ_ANN.nc")

if (
    not (np.max(stj_metrics(metrics=0)[:] - stj_metric_validation_data.STJ_SH[:]))
    < 1e-10
    and (np.max(stj_metrics(metrics=1)[:] - stj_metric_validation_data.STJ_NH[:]))
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated STJ metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated STJ metrics are the same!")


# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
olr_data = pyg.open(data_dir / "rlnt.nc")

# make the axes into the right format
olr_data = olr_data.replace_axes(time=time_axis)

# yearly mean of data
olr_annual = olr_data.rlnt.yearlymean()

olr_metrics = pyt.pyg_olr(olr_annual)

olr_metric_validation_data = pyg.open(metrics_dir / "OLR_ANN.nc")

if (
    not (np.max(olr_metrics(metrics=0)[:] - olr_metric_validation_data.OLR_SH[:]))
    < 1e-10
    and (np.max(olr_metrics(metrics=1)[:] - olr_metric_validation_data.OLR_NH[:]))
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated OLR metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated OLR metrics are the same!")

# P minus E
# read zonal mean monthly precipitation pr(time,lat)
pr_data = pyg.open(data_dir / "pr.nc")
surf_latent_heat_flux_data = pyg.open(data_dir / "hfls.nc")

# make the axes into the right format
pr_data = pr_data.replace_axes(time=time_axis)
surf_latent_heat_flux_data = surf_latent_heat_flux_data.replace_axes(
    time=time_axis, lat=pyg.Lat(surf_latent_heat_flux_data.lat[:])
)

# Latent heat of vaporization
L = 2510400.0

er = -surf_latent_heat_flux_data.hfls / L
pe_data = (pr_data.pr - er).rename("pe")

# yearly mean of data
pe_annual = pe_data.yearlymean()

pe_metrics = pyt.pyg_pe(pe_data)

pe_metric_validation_data = pyg.open(metrics_dir / "PE_ANN.nc")

if (
    not (np.max(pe_metrics(metrics=0)[:] - pe_metric_validation_data.PE_SH[:])) < 1e-10
    and (np.max(pe_metrics(metrics=1)[:] - pe_metric_validation_data.PE_NH[:])) < 1e-10
):
    print("Warning: annual-mean Validation and calculated PE metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated PE metrics are the same!")

# Surface winds
# read zonal mean surface wind U(time,lat)
uas_data = pyg.open(data_dir / "uas.nc")

# Define a time axis
uas_data = uas_data.replace_axes(time=time_axis)

# yearly mean of data
uas_annual = uas_data.uas.yearlymean()

uas_metrics = pyt.pyg_uas(uas_annual)

uas_metric_validation_data = pyg.open(metrics_dir / "UAS_ANN.nc")

if (
    not (np.max(uas_metrics(metrics=0)[:] - uas_metric_validation_data.UAS_SH[:]))
    < 1e-10
    and (np.max(uas_metrics(metrics=1)[:] - uas_metric_validation_data.UAS_NH[:]))
    < 1e-10
):
    print("Warning: annual-mean Validation and calculated UAS metrics are NOT equal!")
else:
    print("OK. Annual-mean Validation and calculated UAS metrics are the same!")
