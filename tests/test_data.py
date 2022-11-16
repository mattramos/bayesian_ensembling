import numpy as np
import xarray as xr
import pytest
from ensembles.data import ModelCollection, ProcessModel
from ensembles.load_data import create_synthetic_data
import pandas as pd
import pytest
import cf_xarray as cfxr

@pytest.mark.parametrize('n_dims', [1,2,3])
def test_process_model(n_dims):
    # Check initialisation
    data_array = create_synthetic_data(n_dims)
    model = ProcessModel(data_array, 'model_name')

    # Check dimensions are correct
    assert model.model_data.ndim == n_dims + 1

    # Check mean, std, max_val and min_val output size
    assert np.all(model.mean_across_realisations.values == model.model_data.mean(dim='realisation').values)
    assert np.all(model.std_across_realisations.values == model.model_data.std(dim='realisation').values)
    assert model.max_val.size == 1
    assert model.min_val.size == 1

    # Check for climatology calculation for dates and given climatology
    model_anomaly = model.calculate_anomaly()
    climatology = model_anomaly.climatology
    model_anomaly = model.calculate_anomaly(climatology=climatology)

    assert model_anomaly.model_data.shape == model.model_data.shape

    # Check looping through realisations
    n_reals = 0
    for realisation in model:
        assert isinstance(realisation, xr.DataArray)
        n_reals += 1
    assert n_reals == model.n_realisations

    model.plot()

@pytest.mark.parametrize('n_dims', [1,2,3])
@pytest.mark.parametrize('n_models', [1,2,3])
def test_model_collection(n_dims, n_models):
    models = []
    for i in range(n_models):
        data_array = create_synthetic_data(n_dims)
        model = ProcessModel(data_array, 'model_name')
        models.append(model)
    model_collection = ModelCollection(models)

    # Check dimension of properties
    assert model_collection.max_val.size == 1
    assert model_collection.min_val.size == 1
    assert model_collection.number_of_models == n_models
    assert len(model_collection.model_names) == n_models

    # Check iterability
    idx = 0
    for model in model_collection:
        assert isinstance(model, ProcessModel)
        idx += 1
    assert idx == n_models

    # Check plotting
    model_collection.plot_all()
    model_collection.plot_grid()

@pytest.mark.parametrize('lat_name', ['lat', 'latitude', 'Latitude'])
@pytest.mark.parametrize('lon_name', ['lon', 'longitude', 'Longitude'])
def test_cf_xarray(lat_name, lon_name):
    n_dims = 3
    da = create_synthetic_data(n_dims)

    # Check that lat and lon coordinates are in da (using cf_xarray)
    assert isinstance(da.cf.mean(lat_name), xr.DataArray)
    assert isinstance(da.cf.mean(lon_name), xr.DataArray)


