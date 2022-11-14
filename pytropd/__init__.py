# cleaning up namespace - sjs 1.27.22
__version__ = "2.0"

# let's only use these modules if the user has installed them
try:
    import xarray

    xarray_installed = True
except ImportError:
    xarray_installed = False

try:
    import pygeode

    pygeode_installed = True
except ImportError:
    pygeode_installed = False

from .metrics import (
    TropD_Metric_EDJ,
    TropD_Metric_OLR,
    TropD_Metric_PE,
    TropD_Metric_PSI,
    TropD_Metric_PSL,
    TropD_Metric_STJ,
    TropD_Metric_TPB,
    TropD_Metric_UAS,
    Shah_2020_GWL,
    Shah_2020_1sigma,
)
from .functions import (
    TropD_Calculate_MaxLat,
    TropD_Calculate_Mon2Season,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_TropopauseHeight,
    TropD_Calculate_ZeroCrossing,
)

if pygeode_installed:
    from .pygeode_metrics import pyg_edj
    from .pygeode_metrics import pyg_olr
    from .pygeode_metrics import pyg_pe
    from .pygeode_metrics import pyg_psl
    from .pygeode_metrics import pyg_uas
    from .pygeode_metrics import pyg_psi
    from .pygeode_metrics import pyg_stj
    from .pygeode_metrics import pyg_tpb
if xarray_installed:
    from .xarray_metrics import MetricAccessor
