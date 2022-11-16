from .array_types import *
from .data import ModelCollection, ProcessModel, Distribution, Observation
from .dtwa import performDBA
from .ensemble_scheme import Barycentre, MultiModelMean, WeightedModelMean
from .models import MeanFieldApproximation, GPDTW1D
from .weights import *

from jax import config
import xarray as xr

xr.set_options(keep_attrs=True)
config.update("jax_enable_x64", True)

__version__ = "0.0.1"
