import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import distrax
import jax.numpy as jnp
import warnings

tfd = tfp.distributions

def sqrtm(A):
    """Fast computation of the matrix square root of matrix A."""
    u, s, v = jnp.linalg.svd(A)
    return u @ jnp.diag(jnp.sqrt(s)) @ v

def wasserstien_distance(A, B):
    """Compute the Wasserstein distance between two distributions described by their covariances A and B."""
    Root_1= sqrtm(A)
    Root_2= sqrtm(B)
    return jnp.trace(A) + jnp.trace(B) - 2*jnp.trace(sqrtm(Root_1 @ B @ Root_1))

def gaussian_w2_distance_distrax(alpha: distrax.Distribution, beta: distrax.Distribution, full_cov=True) -> jnp.ndarray:
    """JAX implementation of the Wasserstein distance between two Gaussian distributions.

    Args:
        alpha (distrax.Distribution): distribution 1
        beta (distrax.Distribution): distribution 2
        full_cov (bool, optional): Whether or not to use the full covariance. Defaults to True.

    Returns:
        jnp.ndarray: Wasserstein distance
    """
    if full_cov:
        mu1, sigma1 = alpha.mean(), alpha.covariance()
        mu2, sigma2 = beta.mean(), beta.covariance()
    else:
        mu1, sigma1 = alpha.mean(), alpha.variance()
        mu2, sigma2 = beta.mean(), beta.variance()
        sigma1 = jnp.diag(sigma1)
        sigma2 = jnp.diag(sigma2)
    location_gap = jnp.linalg.norm(mu1 - mu2, ord=2)
    sigma1_sqrt = sqrtm(sigma1)
    covariance_gap = (
        sigma1 + sigma2 - 2 * sqrtm(sigma1_sqrt @ sigma2 @ sigma1_sqrt)
    )
    w2 = location_gap + jnp.trace(covariance_gap)

    return w2

def gaussian_w2_distance(alpha: tfd.Distribution, beta: tfd.Distribution) -> np.ndarray:
    mu1, sigma1 = alpha.mean(), alpha.covariance()
    mu2, sigma2 = beta.mean(), beta.covariance()
    location_gap = tf.norm(mu1 - mu2, ord=2)
    sigma1_sqrt = tf.linalg.sqrtm(sigma1)
    covariance_gap = (
        sigma1 + sigma2 - 2 * tf.linalg.sqrtm(sigma1_sqrt @ sigma2 @ sigma1_sqrt)
    )
    w2 = location_gap + tf.linalg.trace(covariance_gap)
    return w2.numpy()


def gaussian_barycentre(
    means,
    std_devs,
    weights,
    tolerance: float = 1e-6,
    init_var=1.0,
):
    """Find the Wasserstein barycentre of a set of 1D Gaussian distributions described by their means and standard deviations.

    Args:
        means (jnp.array): The means of the distributions (n_distributions)
        std_devs (jnp.array): The standard deviations of the distributions (n_distributions)
        weights (jnp.array): The weights of the distributions (n_distributions)
        tolerance (float, optional): Tolerance of convergence. Defaults to 1e-6.
        init_var (float, optional): Defaults to 1.0.

    Returns:
        mu, sigma (jnp.array, jnp.array): The mean and standard deviation of the barycentre
    """
    barycentre_variance = init_var
    n_iters = 0
    while True:
        candidate_variance = 0

        for w, s in zip(weights, std_devs):
            candidate_variance += w * jnp.sqrt(barycentre_variance) * s

        if candidate_variance - barycentre_variance < tolerance:
            barycentre_variance = candidate_variance
            break
        else:
            barycentre_variance = candidate_variance
        n_iters += 1
        if n_iters > 200:
            warnings.warn("Barycentre not converged for 1 time step")
            print(f'Difference between variances = {candidate_variance - barycentre_variance}')
            break
    mu = jnp.sum(weights * means)
    sigma = jnp.sqrt(barycentre_variance)
    return mu, sigma
