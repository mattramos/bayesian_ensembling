from _pytest.mark import param
from numpy.lib.arraysetops import isin
import tensorflow as tf
import dataclasses
import gpflow
from gpflow import kernels
import numpy as np
import ensembles
import pytest


@pytest.fixture
def data():
    X = np.random.randn(10, 1)
    Xtest = np.random.randn(50, 1)
    y = np.cos(X)
    return X, Xtest, y


def test_GPR(data):
    X, Xtest, y = data
    n = Xtest.shape[0]
    d = y.shape[1]
    k = gpflow.kernels.RBF()
    m = ensembles.ConjugateGP(kernel=k)
    params = ensembles.config.GPRParameters().to_dict()
    m.fit(X, y, params)
    assert isinstance(m.model, gpflow.models.GPR)

    mu, sigma = m.predict(Xtest, params)
    for mat in [mu, sigma]:
        assert isinstance(mat, tf.Tensor)
        assert mat.numpy().shape == (n, d)


def test_SGPR(data):
    X, Xtest, y = data
    n = Xtest.shape[0]
    d = y.shape[1]
    k = gpflow.kernels.RBF()
    m = ensembles.SparseGP(kernel=k)
    params = ensembles.config.SGPRParameters().to_dict()
    params["n_inducing"] = 5
    params["optim_nits"] = 10
    m.fit(X, y, params)
    assert isinstance(m.model, gpflow.models.SGPR)
    assert isinstance(
        m.model.inducing_variable, gpflow.inducing_variables.InducingPoints
    )

    mu, sigma = m.predict(Xtest, params)
    for mat in [mu, sigma]:
        assert isinstance(mat, tf.Tensor)
        assert mat.numpy().shape == (n, d)
