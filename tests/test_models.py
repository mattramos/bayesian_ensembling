from _pytest.mark import param
from numpy.lib.arraysetops import isin
import tensorflow as tf
import tensorflow_probability as tfp
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


@pytest.mark.parametrize("n_samples", [1, 2, 10])
@pytest.mark.parametrize("d", [1, 2, 10])
def test_joint_reconstruction(d, n_samples):
    mu = np.cos(np.linspace(0, np.pi, num=d))
    x_idx = np.linspace(-3.0, 3.0, num=d).reshape(-1, 1)
    Sigma = gpflow.kernels.Matern32(lengthscales=0.5).K(x_idx)
    L = np.linalg.cholesky(Sigma)
    mvn = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=L)
    marginals = [
        tfp.distributions.Normal(loc=m, scale=s)
        for m, s in zip(mu, np.diag(Sigma))
    ]

    model = ensembles.JointReconstruction(mu, np.diag(Sigma))

    # Check model initialising is being done correctly
    assert isinstance(model.mu, tfp.util.TransformedVariable)
    assert isinstance(model.sigma_hat, tfp.util.TransformedVariable)

    # Check fit
    params = ensembles.config.ReconstructionParameters().to_dict()
    params["optim_nits"] = 5

    y_sample = mvn.sample(n_samples)
    model.fit(y_sample, params)

    # Check returned params
    learned_mu, learned_sigma = model.return_parameters()
    assert isinstance(learned_mu, tf.Tensor)
    assert isinstance(learned_sigma, tf.Tensor)
