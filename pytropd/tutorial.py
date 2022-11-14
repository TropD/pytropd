import numpy as np
from scipy.io import netcdf
import os


def buildV():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../ValidationData/va.nc")
    # 1) PSI -- Streamfunction zero crossing
    # read meridional velocity V(time,lat,lev), latitude and level
    f_V = netcdf.netcdf_file(filename, "r")
    V = f_V.variables["va"][:]
    # Change axes of V to be [time, lat, lev]
    V = np.transpose(V, (2, 1, 0))
    V = V[0, :, :]
    lat = f_V.variables["lat"][:]
    lev = f_V.variables["lev"][:]
    f_V.close()

    return lat, lev, V


lat, lev, V = buildV()
