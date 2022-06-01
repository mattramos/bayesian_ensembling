from .array_types import *
from .data import ModelCollection, ProcessModel, Distribution
from .dtwa import performDBA
from .ensemble_scheme import Barycentre, MultiModelMean, WeightedModelMean
from .ensembles import Ensemble
from .models import MeanFieldApproximation, GPDTW1D, GPDTW3D
from .weights import LogLikelihoodWeight, InverseSquareWeight, UniformWeight

from jax import config

config.update("jax_enable_x64", True)

__version__ = "0.0.1"