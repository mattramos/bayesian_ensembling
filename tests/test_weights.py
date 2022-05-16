from pydoc import cli
import numpy as np
import xarray as xr
import pytest
import ensembles
from ensembles.data import ModelCollection, ProcessModel
from ensembles.weights import InverseSquareWeight, UniformWeight, LogLikelihoodWeight
import pandas as pd
import pytest
import jax.numpy as jnp

def create_xarray_dataarray(n_dims, n_reals=3):
    realisation = np.arange(3)
    time = pd.date_range('1960-01-01', periods=480, freq='M')
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

    
@pytest.mark.parametrize('model_n_dims', [1,2,3])
@pytest.mark.parametrize('obs_n_reals', [1,5,10])
def test_weightings(model_n_dims, obs_n_reals):
    # Check initialisation
    models = []
    n_models = 4
    for i in range(n_models):
        model_data_array = create_xarray_dataarray(model_n_dims)
        model = ProcessModel(model_data_array, 'model_name')
        models.append(model)
    model_collection = ModelCollection(models)

    obs_data_array = create_xarray_dataarray(model_n_dims, obs_n_reals)
    observations = ProcessModel(obs_data_array, 'obs')

    mse_weights = InverseSquareWeight()(model_collection, observations)
    assert isinstance(mse_weights, xr.DataArray)
    assert mse_weights.shape == (n_models, ) + obs_data_array.mean('realisation').shape
    assert np.all(mse_weights.sum('model') == pytest.approx(1., 1e-6))

    uniform_weights = UniformWeight()(model_collection, observations)
    assert isinstance(uniform_weights, xr.DataArray)
    assert uniform_weights.shape == (n_models, ) + obs_data_array.mean('realisation').shape
    assert np.all(uniform_weights.sum('model') == pytest.approx(1., 1e-6))

    # ll_weights, lls = LogLikelihoodWeight()(model_collection, observations, return_lls=True)


