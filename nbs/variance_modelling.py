# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.8.12 ('bayesian_ensembles')
#     language: python
#     name: python3
# ---

# %%
from distutils.command.build import build
import typing as tp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
import pandas as pd
import gpflow
from tqdm import trange
from dataclasses import dataclass


# %%
@dataclass
class Dataset:
    X: np.ndarray
    mu: np.ndarray = None
    std_dev: np.ndarray = None

    def __post_init__(self):
        """Standardise array column-wise"""
        self.mu = np.mean(self.X, axis=0) if not self.mu else self.mu
        self.std_dev = np.std(self.X, axis=0) if not self.std_dev else self.std_dev
        self.Xstd = (self.X - self.mu) / self.std_dev
        self.Xstd = self.Xstd.reshape(-1, 1)


# %%
def load_data(path: str, n_train: int, n_test: int):
    df = pd.read_csv(path)
    realisations = df.values[:, 1:].astype(np.float64)
    # realisations = (realisations - np.mean(realisations, axis=0))/np.std(realisations, axis=0)
    train = realisations[:n_train, :]
    test = realisations[n_train : (n_train + n_test), :]
    return train, test


# %%
def build_kernel(n_years: float):
    base_kernel = gpflow.kernels.Matern32()
    periodic_kernel = gpflow.kernels.Periodic(base_kernel)
    long_kernel = gpflow.kernels.SquaredExponential()
    short_kernel = gpflow.kernels.RationalQuadratic()
    # gpflow.kernels.ArcCosine
    # gpflow.utilities.set_trainable(periodic_kernel.period, False)
    process_kernel = periodic_kernel + long_kernel + short_kernel
    process_kernel.kernels[0].period.assign(3.5 / n_years)  # This is roughly 365 days
    process_kernel.kernels[1].lengthscales.assign(
        30 * 3.5 / n_years
    )  # Roughly a climatological lengthscale
    process_kernel.kernels[2].lengthscales.assign(
        0.5 * 3.5 / n_years
    )  # Short, seasonal lengthscale
    return process_kernel


# %%
def build_model(data: tp.Tuple[np.ndarray, np.ndarray], n_inducing: int):
    n_output_dim = data[1].shape[1]
    n_years = data[0].shape[0] / 265
    kern_list = [build_kernel(n_years) for _ in range(n_output_dim)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    Z = np.linspace(data[0].min(), data[0].max(), n_inducing)[:, None]
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(Z)
    )
    model = gpf.models.SVGP(
        kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2
    )
    return model


# %%
def fit(data, model: gpf.models.model.BayesianModel, n_iters: int, batch_size: int = 256):
    def opt_step(opt, loss, variables):
        opt.minimize(loss, var_list=variables)

    N = data[0].shape[0]
    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
    adam_opt = tf.optimizers.Adam(0.01)

    autotune = tf.data.experimental.AUTOTUNE
    data_minibatch = (
        tf.data.Dataset.from_tensor_slices(data)
        .prefetch(autotune)
        .repeat()
        .shuffle(N)
        .batch(batch_size)
    )
    data_minibatch_it = iter(data_minibatch)

    loss = model.training_loss_closure(data_minibatch_it, compile=True)
    adam_params = model.trainable_variables
    natgrad_params = [(model.q_mu, model.q_sqrt)]

    tr = trange(n_iters)
    for epoch in tr:
        opt_step(adam_opt, loss, adam_params)
        opt_step(natgrad_opt, loss, natgrad_params)

        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % 50 == 0 or epoch == 0:
            likelihood = loss()
            tr.set_postfix({"ELBO": likelihood.numpy()})

    return model


# %%
if __name__ == "__main__":
    n_train = 10000
    n_test = 5000
    n_inducing = 200
    n_optim_nits = 400
    batch_size = 2000

    train, test = load_data(
        "/home/pindert2/bayesian_ensembling/nbs/data/SingLoc_CESM.csv",
        n_train=n_train,
        n_test=n_test,
    )
    Mu = Dataset(X=np.mean(train, axis=1).reshape(-1, 1).astype(np.float64))
    Sigma = Dataset(np.std(train, axis=1).reshape(-1, 1).astype(np.float64))

    Ydata = np.hstack((Mu.Xstd, Sigma.Xstd))
    X = Dataset(np.arange(Ydata.shape[0]).reshape(-1, 1))
    Xdata = X.Xstd
    data = (Xdata, Ydata)

    model = build_model(data, n_inducing=n_inducing)
    model = fit(data, model, n_optim_nits, batch_size=batch_size)

    MuTest = Dataset(
        X=np.mean(test, axis=1).reshape(-1, 1).astype(np.float64), mu=Mu.mu, std_dev=Mu.std_dev
    )
    SigmaTest = Dataset(
        X=np.std(test, axis=1).reshape(-1, 1).astype(np.float64), mu=Sigma.mu, std_dev=Sigma.std_dev
    )

    Xtest = Dataset(
        X=np.arange(n_train, (n_train + n_test)).reshape(-1, 1).astype(np.float64),
        mu=X.mu,
        std_dev=X.std_dev,
    )

    mu_pred, sigma_pred = model.predict_y(Xtest.Xstd)
    mu_pred = mu_pred.numpy()
    sigma_pred = sigma_pred.numpy()

# %%
# fig, ax = plt.subplots(figsize=(16, 16), nrows=2)
# ax[0].plot(X.X, Mu.X, color="tab:blue")
# ax[0].fill_between(X.X.squeeze(), Mu.X.squeeze() - Sigma.X.squeeze(), Mu.X.squeeze() + Sigma.X.squeeze(), color="tab:blue", alpha=0.3)
# ax[0].plot(Xtest.X, np.mean(test, axis=1), color='tab:blue')
# ax[0].plot(Xtest.X, mu_pred[:, 0], color='tab:orange')
# ax[0].fill_between(Xtest.X.squeeze(), mu_pred[:, 0] - np.sqrt(sigma_pred[:, 0]), mu_pred[:, 0] + np.sqrt(sigma_pred[:, 0]), color='tab:orange', alpha=0.3)

# ax[1].plot(X.X, Sigma.X, color="tab:blue")
# # ax[1].fill_between(X.X.squeeze(), Sigma.X.squeeze() - Sigma.X.squeeze(), Mu.X.squeeze() + Sigma.X.squeeze(), color="tab:blue", alpha=0.3)
# # ax[1].plot(Xtest.X, np.mean(test, axis=1), color='tab:blue')
# ax[1].plot(Xtest.X, mu_pred[:, 1], color='tab:orange')
# # ax[1].fill_between(Xtest.X.squeeze(), mu_pred[:, 0] - np.sqrt(sigma_pred[:, 0]), mu_pred[:, 0] + np.sqrt(sigma_pred[:, 0]), color='tab:orange', alpha=0.3)

xidx = X.X
Xte_idx = Xtest.X

fig, ax = plt.subplots(figsize=(16, 12), nrows=2)
ax[0].plot(xidx, Mu.Xstd, color='tab:blue')
ax[0].fill_between(xidx.squeeze(), Mu.Xstd.squeeze() - Sigma.Xstd.squeeze(), Mu.Xstd.squeeze() + Sigma.Xstd.squeeze(), color="tab:blue", alpha=0.3)
ax[0].plot(Xte_idx, mu_pred[:, 0], color='tab:orange')
ax[0].fill_between(Xte_idx.squeeze(), mu_pred[:, 0] - np.sqrt(sigma_pred[:, 0]), mu_pred[:, 0] + np.sqrt(sigma_pred[:, 0]), color='tab:orange', alpha=0.3)

ax[1].plot(xidx, Sigma.Xstd, color='tab:blue', alpha=0.5)
ax[1].plot(Xte_idx, mu_pred[:, 1], color='tab:orange')
ax[1].fill_between(Xte_idx.squeeze(), mu_pred[:, 1] - np.sqrt(sigma_pred[:, 1]), mu_pred[:, 1] + np.sqrt(sigma_pred[:, 1]), color='tab:orange', alpha=0.3)




# %%
Xte_idx

# %%
