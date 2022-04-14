from .config import GPRParameters, SGPRParameters
from .data import Dataset, ProcessModel
from .ensembles import Ensemble
from .models import (
    ConjugateGP,
    GPFlowModel,
    Model,
    SparseGP,
    JointReconstruction,
)
from .utils import simulate_data
from .hgp import HSGP
from .plotters import plot_realisations, plot_individual_preds
from .wasserstein import gaussian_barycentre, gaussian_w2_distance

__version__ = "0.0.1"
