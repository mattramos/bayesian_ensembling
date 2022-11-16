import numpy as np
import xarray as xr
import pandas as pd
from ensembles.data import ModelCollection, ProcessModel, Observation


def create_synthetic_data(n_dims=3, n_lats=5, n_lons=4, n_time=24, n_realisations=3):
    realisation = np.arange(n_realisations)
    time = pd.date_range('1960-01-01', periods=n_time, freq='M')
    lat = np.arange(n_lats)
    lon = np.arange(n_lons)

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
        model_data_array = create_synthetic_data(n_dims=model_n_dims)
        model = ProcessModel(model_data_array, f'model{i}')
        models.append(model)
    model_collection = ModelCollection(models)

    obs_data_array = create_synthetic_data(n_dims=model_n_dims, n_reals=obs_n_reals)
    observations = Observation(obs_data_array)

    return model_collection, observations