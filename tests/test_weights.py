from pydoc import cli
import numpy as np
import xarray as xr
import pytest
import ensembles as es
from ensembles.data import ModelCollection, ProcessModel
from ensembles.weights import *
import pandas as pd
import pytest
import jax.numpy as jnp

def create_xarray_dataarray(n_dims, n_reals=3):
    """Create a xarray DataArray for testing with n_dims dimensions and n_reals realisations.
    This is a helper function for creating test data."""
    realisation = np.arange(n_reals)
    time = pd.date_range('1960-01-01', periods=24, freq='M')
    lat = np.arange(5)
    lon = np.arange(4)

    ds = xr.Dataset(
        data_vars=dict(
            val=(
                ['realisation', 'time', 'lon', 'lat'],
                np.random.rand(
                    len(realisation),
                    len(time),
                    len(lon),
                    len(lat)
                )
            )
        ),
        coords=dict(
            realisation=(['realisation'], realisation),
            time=(['time'], time),
            lon=(['lon'], lon),
            lat=(['lat'], lat),
        )
    )

    if n_dims == 1:
        return ds.val.isel(lat=0, lon=0).drop_vars(['lat', 'lon'])
    elif n_dims == 2:
        return ds.val.isel(lat=0,).drop_vars(['lat'])
    elif n_dims == 3:
        return ds.val

def create_synthetic_models_and_obs(n_models=3, obs_n_reals=3, model_n_dims=3):
    """Create synthetic models and observations for testing.

    Args:
        n_models (int, optional): Number of models (process). Defaults to 3.
        obs_n_reals (int, optional): Number of observational realisations. Defaults to 3.
        model_n_dims (int, optional): Number of model dimensions (time, lat, lon). Defaults to 3.

    Returns:
        ModelCollection, ProcessModel: the model collection and the observations
    """

    models = []
    for i in range(n_models):
        model_data_array = create_xarray_dataarray(n_dims=model_n_dims)
        model = ProcessModel(model_data_array, f'model{i}')
        models.append(model)
    model_collection = ModelCollection(models)

    obs_data_array = create_xarray_dataarray(n_dims=model_n_dims, n_reals=obs_n_reals)
    observations = ProcessModel(obs_data_array, 'obs')

    return model_collection, observations
    
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
