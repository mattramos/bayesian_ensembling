from pydoc import cli
import numpy as np
import xarray as xr
import pytest
import ensembles
from ensembles.data import ProcessModel
import pandas as pd
import pytest

def create_xarray_dataarray(n_dims):
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

@pytest.mark.parametrize('n_dims', [1,2,3])
def test_process_model(n_dims):
    # Check initialisation
    data_array = create_xarray_dataarray(n_dims)
    model = ProcessModel(data_array, 'model_name')

    # Check dimensions are correct
    assert model.model_data.ndim == n_dims + 1

    # Check mean, std, max_val and min_val output size
    assert model.model_mean.size == 1
    assert model.model_std.size == 1
    assert model.max_val.size == 1
    assert model.min_val.size == 1

    # Check standardisation
    standardised_model = model.standardise_data()
    assert standardised_model.model_mean == pytest.approx(0, 1e-6)
    assert standardised_model.model_std == pytest.approx(1, 1e-6)
    assert isinstance(standardised_model, ProcessModel)

    # Check unstandardisation
    unstandardised_model = standardised_model.unstandardise_data()
    assert np.all(model.model_data == pytest.approx(unstandardised_model.model_data, 1e-6))
    assert isinstance(unstandardised_model, ProcessModel)

    # Check for climatology calculation for dates and given climatology
    model_anomaly = model.calculate_anomaly()
    climatology = model_anomaly.climatology
    model_anomaly = model.calculate_anomaly(climatology=climatology)

    assert model_anomaly.model_data.shape == model.model_data.shape

    # Check looping through realisations
    n_reals = 0
    for realisation in model:
        n_reals += 1
    assert n_reals == model.n_realisations

@pytest.mark.parametrize('n_dims', [1,2,3])
@pytest.mark.parametrize('n_models', [1,2,3])
def test_model_collection(n_dims):
    print()


