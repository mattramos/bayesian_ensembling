import numpy as np
import ot
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions


def gaussian_w2_distance(alpha: tfd.Distribution, beta: tfd.Distribution) -> np.ndarray:
    mu1, sigma1 = alpha.mean(), alpha.covariance()
    mu2, sigma2 = beta.mean(), beta.covariance()
    location_gap = tf.norm(mu1 - mu2, ord=2)
    sigma1_sqrt = tf.linalg.sqrtm(sigma1)
    covariance_gap = sigma1 + sigma2 - 2 * tf.linalg.sqrtm(sigma1_sqrt @ sigma2 @ sigma1_sqrt)
    w2 = location_gap + tf.linalg.trace(covariance_gap)
    return w2.numpy()


def gaussian_barycentre(
    means,
    std_devs,
    weights,
    tolerance: float = 1e-6,
    init_var=1.0,
    as_hist: bool = False,
    n_bins=100,
):
    barycentre_variance = init_var
    while True:
        candidate_variance = 0

        for w, s in zip(weights, std_devs):
            candidate_variance += w * np.sqrt(barycentre_variance) * s

        if candidate_variance - barycentre_variance < tolerance:
            barycentre_variance = candidate_variance
            break
        else:
            barycentre_variance = candidate_variance
    mu = np.sum(weights * means)
    sigma = np.sqrt(barycentre_variance)
    if as_hist:
        return ot.datasets.make_1D_gauss(n_bins, mu, sigma)
    else:
        return mu, sigma


def mvgaussian_barycentre(
    means,
    covariances,
    weights,
    tolerance: float = 1e-6,
    init_var=1.0,
    as_hist: bool = False,
    n_bins=100,
):
    barycentre_variance = init_var
    while True:
        candidate_variance = 0

        for w, s in zip(weights, std_devs):
            candidate_variance += w * np.sqrt(barycentre_variance) * s

        if candidate_variance - barycentre_variance < tolerance:
            barycentre_variance = candidate_variance
            break
        else:
            barycentre_variance = candidate_variance
    mu = np.sum(weights * means)
    sigma = np.sqrt(barycentre_variance)
    if as_hist:
        return ot.datasets.make_1D_gauss(n_bins, mu, sigma)
    else:
        return mu, sigma


def print_numbers(numbers):
    for n in numbers:
        print(n)
