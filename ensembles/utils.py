from ast import Return
from parso import parse
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gpflow
import typing as tp
from argparse import ArgumentParser


def simulate_data(
    n_obs: int,
    n_realisations: int,
    noise_lims: tp.Tuple[float, float],
    true_kernel: gpflow.kernels.Kernel,
    xlims: tp.Tuple[float, float] = (-5.0, 5.0),
    jitter_amount: float = 1e-8,
    seed_value: int = 123,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    tfp_seed = tfp.random.sanitize_seed(seed_value)
    rng = np.random.RandomState(123)
    X = np.sort(rng.uniform(*xlims, (n_obs, 1)), axis=0)
    true_kernel = gpflow.kernels.Matern32()
    Kxx = true_kernel(X) + tf.cast(tf.eye(n_obs) * jitter_amount, dtype=tf.float64)
    latent_y = tfp.distributions.MultivariateNormalTriL(
        np.zeros(n_obs), tf.linalg.cholesky(Kxx)
    ).sample(seed=tfp_seed)

    noise_terms = np.random.uniform(*noise_lims, size=n_realisations)
    realisations = []

    for noise in noise_terms:
        sample_y = latent_y.numpy() + rng.normal(
            loc=0.0, scale=noise, size=latent_y.numpy().shape
        )
        realisations.append(sample_y)
    Y = np.asarray(realisations).T
    return X, latent_y, Y
