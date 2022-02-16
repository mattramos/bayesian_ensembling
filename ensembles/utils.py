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
        sample_y = latent_y.numpy() + rng.normal(loc=0.0, scale=noise, size=latent_y.numpy().shape)
        realisations.append(sample_y)
    Y = np.asarray(realisations).T
    return X, latent_y, Y


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--n_obs", type=int, default=100, help="Number of observations per realisation"
    )
    parser.add_argument(
        "-r", "--n_realisation", type=int, default=5, help="Number of realisations."
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.5,
        help=(
            "Upper limit on the amount of noise that can be added to individual realisations'"
            " points."
        ),
    )
    parser.add_argument(
        "--xlims",
        type=float,
        default=5.0,
        help="Upper and lower x-axis limit.",
    )
    parser.add_argument(
        "--n_inducing",
        type=int,
        default=50,
        help="Number of inducing points used in the hierarchical GP.",
    )
    parser.add_argument(
        "--optim_nits",
        type=int,
        default=50,
        help="Number of optimisation steps to be carried out.",
    )
    parser.add_argument(
        "-k",
        "--true_kernel",
        type=str,
        default="RBF",
        choices=["Matern12", "Matern32", "Matern52", "RBF"],
        help="The true kernel from which realisations should be drawn from.",
    )
    parser.add_argument(
        "--true_lengthscale",
        type=float,
        default=1.0,
        help="The true kernel lengthscale.",
    )
    parser.add_argument(
        "--true_variance",
        type=float,
        default=1.0,
        help="The true kernel variance.",
    )
    parser.add_argument(
        "--group_kernel",
        type=str,
        default="RBF",
        choices=["Matern12", "Matern32", "Matern52", "RBF"],
        help="The hierarchical GP's group kernel.",
    )
    parser.add_argument(
        "--individual_kernel",
        type=str,
        default="RBF",
        choices=["Matern12", "Matern32", "Matern52", "RBF"],
        help="The hierarchical GP's individual kernel for each realisation.",
    )
    parser.add_argument("--train_inducing", type=bool, default=False)
    args = vars(parser.parse_args())
    args["true_kernel"] = _str_kernel_map(
        args["true_kernel"], args["true_lengthscale"], args["true_variance"]
    )
    args["individual_kernel"] = _str_kernel_map(args["individual_kernel"])
    args["group_kernel"] = _str_kernel_map(args["group_kernel"])
    return args


def _str_kernel_map(
    kernel_string: str, lengthscale: float = 1.0, variance: float = 1.0
) -> gpflow.kernels.Kernel:
    if kernel_string == "Matern12":
        return gpflow.kernels.Matern12(lengthscales=lengthscale, variance=variance)
    if kernel_string == "Matern32":
        return gpflow.kernels.Matern32(lengthscales=lengthscale, variance=variance)
    if kernel_string == "Matern52":
        return gpflow.kernels.Matern52(lengthscales=lengthscale, variance=variance)
    if kernel_string == "RBF":
        return gpflow.kernels.RBF(lengthscales=lengthscale, variance=variance)
