from .array_types import *
from .data import ModelCollection, ProcessModel
from .ensemble_scheme import Barycentre, MultiModelMean, WeightedModelMean
from .ensembles import Ensemble
from .models import MeanFieldApproximation
from .weights import LogLikelihoodWeight, InverseSquareWeight, UniformWeight

__version__ = "0.0.1"
