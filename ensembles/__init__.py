from .array_types import *
from .data import ModelCollection, ProcessModel
from .dtwa import performDBA
from .ensemble_scheme import Barycentre, MultiModelMean, WeightedModelMean
from .ensembles import Ensemble
from .models import MeanFieldApproximation, FullRankApproximation, GPDTW
from .weights import LogLikelihoodWeight, InverseSquareWeight, UniformWeight

from jax import config

config.update("jax_enable_x64", True)

__version__ = "0.0.1"
