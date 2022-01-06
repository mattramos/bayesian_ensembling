# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''bayesian_ensembles'': conda)'
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np

# +
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter
from gpflow.kernels import Matern32
from tqdm import trange

tfd = tfp.distributions

# +
mu = np.array([0.5, 0.6, 0.8])
Sigma = np.array([[0.9, 0.6, 0.5], [0.6, 0.85, 0.55], [0.5, 0.9, 1.0]])
L = np.linalg.cholesky(Sigma)

mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
marginals = [tfd.Normal(loc=m, scale=s) for m, s in zip(mu, np.diag(Sigma))]


class JointReconstruction(tf.Module):
    def __init__(self, mu: np.ndarray, variances: np.ndarray, name=None):
        # self.mu = tf.cast(mu, dtype=tf.float64)
        self.mu = tfp.util.TransformedVariable(
            initial_value=tf.cast(mu, dtype=tf.float64),
            bijector=tfp.bijectors.Identity(),
            dtype=tf.float64,
            trainable=False,
        )
        self.variances = variances
        self.sigma_hat = tfp.util.TransformedVariable(
            initial_value=tf.cast(
                tf.eye(num_rows=mu.shape[0]) * variances, dtype=tf.float64
            ),
            bijector=tfp.bijectors.FillTriangular(),
            dtype=tf.float64,
            trainable=True,
        )

    def log_likelihood(self, x: np.ndarray):
        dist = tfd.MultivariateNormalTriL(self.mu, self.sigma_hat)
        return -tf.reduce_sum(dist.log_prob(x))


y_sample = mvn.sample(50)
mvn_hat = JointReconstruction(mu, np.diag(Sigma))

opt = tf.optimizers.Adam(learning_rate=0.005)
losses = []
for _ in trange(500):
    with tf.GradientTape() as tape:
        tape.watch(mvn_hat.trainable_variables)
        loss = mvn_hat.log_likelihood(y_sample)
    grads = tape.gradient(loss, mvn_hat.trainable_variables)
    opt.apply_gradients(zip(grads, mvn_hat.trainable_variables))

# +
import gpflow


class GPFlowJointReconstruction(gpflow.models.model.BayesianModel):
    def __init__(self, mu: np.ndarray, variances: np.ndarray, name=None):
        self.mu = gpflow.base.Parameter(
            value=tf.cast(mu.reshape(-1, 1), dtype=tf.float64),
            transform=tfp.bijectors.Identity(),
        )
        self.variances = variances
        self.sigma_hat = gpflow.base.Parameter(
            value=tf.expand_dims(
                tf.cast(
                    tf.eye(num_rows=mu.shape[0]) * variances, dtype=tf.float64
                ),
                axis=0,
            ),
            transform=tfp.bijectors.FillTriangular(),
        )
        super().__init__(name=name)

    def maximum_log_likelihood_objective(self, x: np.ndarray):
        dist = tfd.MultivariateNormalTriL(
            tf.squeeze(self.mu), tf.squeeze(self.sigma_hat)
        )
        return tf.reduce_sum(dist.log_prob(x))

    def log_likelihood(self, x: np.ndarray):
        return -self.maximum_log_likelihood_objective(x)


# -

mvn_hat = GPFlowJointReconstruction(mu=mu, variances=np.diag(Sigma))

y_sample = mvn.sample(50)

mvn_hat.log_likelihood(y_sample)

# +
opt = tf.optimizers.Adam(learning_rate=0.005)
losses = []
for _ in trange(500):
    with tf.GradientTape() as tape:
        tape.watch(mvn_hat.trainable_variables)
        loss = mvn_hat.log_likelihood(y_sample)
    grads = tape.gradient(loss, mvn_hat.trainable_variables)
    opt.apply_gradients(zip(grads, mvn_hat.trainable_variables))
    losses.append(loss)

plt.plot(losses)
# -

mvn_hat.sigma_hat.numpy().round(4)

L.round(4)


# +
target_dim = 5
mu = np.cos(np.linspace(0, np.pi, num=target_dim))
x_idx = np.linspace(-3.0, 3.0, num=target_dim).reshape(-1, 1)
Sigma = Matern32(lengthscales=0.2).K(x_idx)
L = np.linalg.cholesky(Sigma)

mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
marginals = [tfd.Normal(loc=m, scale=s) for m, s in zip(mu, np.diag(Sigma))]
# -

mvn_hat = GPFlowJointReconstruction(mu=mu, variances=np.diag(Sigma))
y_sample = mvn.sample(50)
mvn_hat.log_likelihood(y_sample)

# +
opt = tf.optimizers.Adam(learning_rate=0.001)
losses = []
for _ in trange(500):
    with tf.GradientTape() as tape:
        tape.watch(mvn_hat.trainable_variables)
        loss = mvn_hat.log_likelihood(y_sample)
    grads = tape.gradient(loss, mvn_hat.trainable_variables)
    opt.apply_gradients(zip(grads, mvn_hat.trainable_variables))
    losses.append(loss)

plt.plot(losses)
# -

mvn_hat.sigma_hat.numpy().round(2)

L.round(4)

from gpflow.base import Parameter
from gpflow.optimizers import NaturalGradient

opt = NaturalGradient(gamma=0.01)
mvn_hat = GPFlowJointReconstruction(mu=mu, variances=np.diag(Sigma))

# +
mlls = []


@tf.function
def objective():
    return mvn_hat.log_likelihood(y_sample)


for _ in range(20):
    opt.minimize(objective, [(mvn_hat.mu, mvn_hat.sigma_hat)])
    mlls.append(objective())

plt.plot(mlls)
# -

mvn_hat.sigma_hat.numpy().squeeze().round(3)

L.round(3)

mu

mvn_hat.mu
