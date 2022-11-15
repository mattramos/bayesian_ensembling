from .array_types import *
from .data import ModelCollection, ProcessModel, Distribution
from .dtwa import performDBA
from .ensemble_scheme import Barycentre, MultiModelMean, WeightedModelMean
from .models import MeanFieldApproximation, GPDTW1D
from .weights import *

from jax import config

config.update("jax_enable_x64", True)

__version__ = "0.0.1"
