import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
import pandas as pd
from glob import glob
import gpflow
from tqdm import trange


def build_kernel(n_years):
    base_kernel = gpflow.kernels.Matern32()
    periodic_kernel = gpflow.kernels.Periodic(base_kernel)
    long_kernel = gpflow.kernels.SquaredExponential()
    short_kernel = gpflow.kernels.RationalQuadratic()
    process_kernel = periodic_kernel + long_kernel + short_kernel
    process_kernel.kernels[0].period.assign(3.5 / n_years)  # This is roughly 12 days
    process_kernel.kernels[1].lengthscales.assign(
        30 * 3.5 / n_years
    )  # Roughly a climatological lengthscale
    process_kernel.kernels[2].lengthscales.assign(
        0.5 * 3.5 / n_years
    )  # Short, seasonal lengthscale
    return process_kernel

def monthly_average(df):
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d")
    df_month = df.resample('M', on='time').mean()
    return df_month


if __name__ == "__main__":
    n_years = 10
    n_years_predict = 5
    n_inducing = 100
    epochs = 2500
    log_freq = 50
    model_names = ["OBS", "CESM", "CANESM", "CSIRO", "GFDL-esm2m", "ECEARTH", "ALL"]
    standardisers = {}
    for mn in model_names:
        df = pd.read_csv(f"./data/SingLoc_{mn}.csv")
        if 'obs_mean' in df.columns:
            df.drop(['obs_mean', 'obs_std'], axis=1, inplace=True)

        df = monthly_average(df)

        # Time is an index (caused by the pandas.resample)
        realisations = df.values.astype(np.float64)
        # realisations = (realisations - np.mean(realisations, axis=0))/np.std(realisations, axis=0)
        print(realisations.shape)

        rsubset = realisations[: n_years * 12, :]
        rsubset.shape

        mus = np.mean(rsubset, axis=1).astype(np.float64)
        sigmas = np.std(rsubset, axis=1).astype(np.float64)
        mu_mu = np.mean(mus)
        sigma_mu = np.mean(sigmas)
        mu_sigma = np.std(mus)
        sigma_sigma = np.std(sigmas)
        mus = (mus - mu_mu) / mu_sigma
        sigmas = (sigmas - sigma_mu) / sigma_sigma
        mus *= -1
        xidx = np.arange(mus.shape[0])
        standardisers[mn] = [mu_mu, mu_sigma, sigma_mu, sigma_sigma]

        test_realisations = realisations[(n_years * 12) : (n_years + n_years_predict) * 12, :]
        mur_test = (np.mean(test_realisations, axis=1).astype(np.float64) - mu_mu) / mu_sigma
        sigmar_test = (
            np.std(test_realisations, axis=1).astype(np.float64) - sigma_mu
        ) / sigma_sigma

        # np.mean(rsubset, axis=1).shape
        Y_data = np.hstack((mus.reshape(-1, 1), sigmas.reshape(-1, 1)))
        X_data = np.arange(mus.shape[0]).reshape(-1, 1).astype(np.float64)
        Xtr_mu = np.mean(X_data)
        Xtr_std = np.std(X_data)
        X_data = (X_data - Xtr_mu) / Xtr_std
        data = (X_data, Y_data)
        kern_list = [build_kernel(n_years=n_years) for _ in range(Y_data.shape[1])]
        kernel = gpf.kernels.SeparateIndependent(kern_list)

        Z = np.linspace(X_data.min(), X_data.max(), n_inducing)[
            :, None
        ]  # Z must be of shape [M, 1]

        iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Z)
        )

        model = gpf.models.SVGP(
            kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2
        )
        gpflow.utilities.print_summary(model, fmt="notebook")

        loss_fn = model.training_loss_closure(data)

        gpf.utilities.set_trainable(model.q_mu, False)
        gpf.utilities.set_trainable(model.q_sqrt, False)

        variational_vars = [(model.q_mu, model.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = model.trainable_variables
        adam_opt = tf.optimizers.Adam(0.01)

        @tf.function
        def optimisation_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        tr = trange(epochs)

        for epoch in tr:
            optimisation_step()

            # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
            if epoch % log_freq == 0 or epoch == 0:
                likelihood = model.elbo(data)
                tr.set_postfix({"ELBO": likelihood.numpy()})

        yte = realisations[: (n_years + n_years_predict) * 12, :]
        Xte = np.arange(0, (n_years + n_years_predict) * 12).reshape(-1, 1).astype(np.float64)
        Xte = (Xte - Xtr_mu) / Xtr_std
        Xte_idx = np.arange(0, (n_years + n_years_predict) * 12).astype(np.float64)
        mu_te, sigma_te = model.predict_y(Xte)
        mu_te = mu_te.numpy()
        sigma_te = sigma_te.numpy()

        df = pd.DataFrame(
            data=np.hstack(
                (
                    Xte_idx.reshape(-1, 1),
                    mu_te,
                    sigma_te,
                    np.hstack(
                        (
                            np.vstack((mus.reshape(-1, 1), mur_test.reshape(-1, 1))),
                            np.vstack((sigmas.reshape(-1, 1), sigmar_test.reshape(-1, 1))),
                        ),
                    ),
                    np.repeat(mn, mu_te.shape[0]).reshape(-1, 1),
                )
            ),
            columns=[
                "idx",
                "pred_mu",
                "pred_sigma",
                "mu_variance",
                "sigma_variance",
                "true_mu",
                "true_sigma",
                "model",
            ],
        )

        df.to_csv(f"output/preds/model_output_monthly_{mn}.csv", index=False)

        fig, ax = plt.subplots(figsize=(16, 12), nrows=2)
        ax[0].plot(xidx, mus, color="tab:blue")
        ax[0].fill_between(xidx, mus - sigmas, mus + sigmas, color="tab:blue", alpha=0.3)
        # ax[0].plot(Xte_idx, np.mean(yte, axis=1), color='tab:blue')
        ax[0].plot(Xte_idx, mu_te[:, 0], color="tab:orange")
        ax[0].fill_between(
            Xte_idx,
            mu_te[:, 0] - np.sqrt(sigma_te[:, 0]),
            mu_te[:, 0] + np.sqrt(sigma_te[:, 0]),
            color="tab:orange",
            alpha=0.3,
        )

        ax[1].plot(xidx, sigmas, color="tab:blue", alpha=0.5)
        ax[0].fill_between(xidx, mus - sigmas, mus + sigmas, color="tab:blue", alpha=0.3)
        # ax[0].plot(Xte_idx, np.mean(yte, axis=1), color='tab:blue')
        ax[1].plot(Xte_idx, mu_te[:, 1], color="tab:orange")
        ax[1].fill_between(
            Xte_idx,
            mu_te[:, 1] - np.sqrt(sigma_te[:, 1]),
            mu_te[:, 1] + np.sqrt(sigma_te[:, 1]),
            color="tab:orange",
            alpha=0.5,
        )
        plt.savefig(f"output/figs/monthly_{mn}.png")
    stds = pd.DataFrame().from_dict(standardisers).T.reset_index()
    stds.columns = ["model", "mean_mean", "mean_sd", "sd_mean", "sd_sd"]
    stds.to_csv("output/aux/standardisers_monthly.csv")
