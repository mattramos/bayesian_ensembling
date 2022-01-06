import numpy as np
from ensembles.data import Dataset
import pytest
import ensembles


@pytest.mark.parametrize("n", [1, 2, 5, 10])
@pytest.mark.parametrize("n_data", [1, 2, 5, 10])
def test_dataset(n, n_data):
    X = [np.random.randn(n, 2)] * n_data
    y = np.random.randn(n, 1)
    data = Dataset(X, y)

    # Assert basic properties of the dataset object
    assert data.n == n
    assert len(data) == n
    assert data.n_datasets == n_data

    # Check an error will be raised if input/output datasets are of differing length
    y_false = np.random.randn(n + 1, 1)
    with pytest.raises(ValueError):
        Dataset(X, y_false)

    # Check an error will be raised if the input data is not a list
    with pytest.raises(AssertionError):
        Dataset(X[0], y)
