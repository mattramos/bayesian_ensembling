from pydoc import cli
import numpy as np
import xarray as xr
import pytest
import ensembles as es
from ensembles.data import ModelCollection, ProcessModel
from ensembles.weights import *
from ensembles.load_data import create_synthetic_data, create_synthetic_models_and_obs
import pandas as pd
import pytest
import jax.numpy as jnp
    
@pytest.mark.parametrize('model_n_dims', [1]) # Only test 1D for now
@pytest.mark.parametrize('obs_n_reals', [1,2,5,10])
@pytest.mark.parametrize('n_models', [2,5,10]) # Not testing one as we're looking at ensembles
def test_weightings(model_n_dims, obs_n_reals, n_models):
    """Test that the weighting functions return the correct shape.
    We test all weights for shape, type (xr.DataArray) and that the sum of weights is 1."""

    weightings = [
        InverseSquareWeight(),
        UniformWeight(),
        LogLikelihoodWeight(),
        ModelSimilarityWeight(),
        KSDWeight(),
        CRPSWeight()
    ]

    model_collection, observations = create_synthetic_models_and_obs(n_models=n_models, obs_n_reals=obs_n_reals, model_n_dims=model_n_dims)

    # Fit models
    model_collection.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2, progress_bar=False)

    for weighting_method in weightings:
        if weighting_method.name == 'LogLikelihood': # We standardise here to avoid numerical issues
            weights = weighting_method(model_collection, observations, standardisation_constant=0.001)
        elif weighting_method.name == 'ModelSimilarityWeight':
            weights = weighting_method(model_collection, observations, mode='temporal')
        else:
            weights = weighting_method(model_collection, observations)
        assert isinstance(weights, xr.DataArray)
        assert weights.shape == (n_models, ) + observations.model_data.mean('realisation').shape
        assert np.all(weights.sum('model') == pytest.approx(1., 1e-6))
