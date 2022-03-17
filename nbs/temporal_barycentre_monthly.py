from re import A
import polars as pl
import ot
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
from glob import glob
import seaborn as sns
import subprocess
from scipy.stats import norm
from tqdm import trange, tqdm


def gaussian_barycentre(
    means,
    std_devs,
    weights,
    tolerance: float = 1e-6,
    init_var=1.0,
    as_hist: bool = False,
    x=np.arange(300),
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
        return norm(mu, sigma).pdf(x)
    else:
        return mu, sigma


def tidy_legend(ax):
    """
    Tidy up a plot's legend by removing duplicate entries
    :param ax: The matplotlib axes object where the legend labels reside.
    :return:
    """
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    # Add labels to plot
    ax.legend(handles, labels, loc="best")
    return ax

def monthly_average(df):
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d")
    df_month = df.resample('M', on='time').mean()
    return df_month

def get_naive_est():
    files = glob('data/SingLoc_*.csv')
    # Remove obs from naive mean
    _ = files.remove('data/SingLoc_OBS.csv')
    _ = files.remove('data/SingLoc_ALL.csv')
    dfs = np.hstack([monthly_average(pd.read_csv(file)).values.astype(np.float64) for file in files])
    ens_naive_mean = np.mean(dfs, axis=1)
    ens_naive_std = np.std(dfs, axis=1)
    
    return ens_naive_mean, ens_naive_std


if __name__ == "__main__":
    n_days = 12 * 10
    step = 1
    x = np.linspace(200, 300, 1000)
    build_gaussian = lambda v: norm(v[0], v[1]).pdf(x)

    df_read = pl.concat([pl.read_csv(m)[:n_days] for m in glob("output/preds/*monthly*.csv")])
    df_read['pred_mu'] = df_read['pred_mu'] * -1

    standardisers = pl.read_csv("output/aux/standardisers_monthly.csv")
    df_read = df_read.join(standardisers, on="model", how="left")

    df_read = df_read.with_columns(
    ((pl.col("pred_mu") * pl.col("mean_sd")) + pl.col("mean_mean")).alias("mean")
    )
    df_read = df_read.with_columns(
        ((pl.col("pred_sigma") * pl.col("sd_sd")) + pl.col("sd_mean")).alias("sd")
    )
    n_obs = (
        df_read.groupby("model")
        .agg([pl.col("mean").count()])
        .select(pl.col("mean"))
        .to_numpy()[0][0]
    )
    bary_mus = []
    bary_sigmas = []

    ens_naive_mean, ens_naive_std = get_naive_est()

    for i in tqdm(range(0, n_days, step)):
        df = df_read.melt(id_vars=["idx", "model"], value_vars=["mean", "sd"])
        df = df.filter(pl.col("idx") == i)
        df = df.pivot(values="value", index="model", columns="variable")
        df_models = df.filter(pl.col("model") != "OBS").filter(pl.col("model") != "ALL")
        df_obs = df.filter(pl.col("model") == "OBS")
        df_all = df.filter(pl.col("model") == "ALL")

        marginals = df_models.select(["mean", "sd"]).to_numpy()
        marginal_dists = [build_gaussian(m) for m in marginals]
        marginal_names = [n for n in df_models.to_series(0)]

        obs_dist_vars = df_obs.select(["mean", "sd"]).to_numpy()[0]
        obs_dist = build_gaussian(obs_dist_vars)
        obs_name = "OBS"

        all_dist_vars = df_all.select(["mean", "sd"]).to_numpy()[0]
        all_dist = build_gaussian(all_dist_vars)
        all_name = "ALL"

        naive_dist = build_gaussian([ens_naive_mean[i], ens_naive_std[i]])

        n_distributions = marginals.shape[0]
        means = marginals[:, 0]
        std_devs = marginals[:, 1]
        weights = np.array([1 / n_distributions] * n_distributions)
        bary = gaussian_barycentre(means, std_devs, weights, as_hist=True, x=x)
        bary_mu, bary_sd = gaussian_barycentre(means, std_devs, weights, as_hist=False)
        bary_mus.append(bary_mu)
        bary_sigmas.append(bary_sd)

        bmu_array = np.array(bary_mus)
        bsig_array = np.array(bary_sigmas)

        cols = list(sns.color_palette("tab10"))

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [2, 1]})
        for m, n, c in zip(marginal_dists, marginal_names, cols):
            ax0.plot(x, m, linestyle="--", color=c, alpha=0.4, label=n)
            df_slice = df_read.filter((pl.col("model") == n) & (pl.col("idx") <= i))
            ax1.plot(df_slice.select(pl.col("mean")).to_numpy(), color=c, alpha=0.35, label=n)

        ax0.plot(x, obs_dist, linestyle="-.", color='black', alpha=0.6, label=obs_name)
        ax0.plot(x, all_dist, linestyle="--", color='black', alpha=0.6, label=all_name)
        df_slice = df_read.filter((pl.col("model") == obs_name) & (pl.col("idx") <= i))
        ax1.plot(df_slice.select(pl.col("mean")).to_numpy(), linestyle="-.", color='black', alpha=0.35, label=obs_name)
        df_slice = df_read.filter((pl.col("model") == all_name) & (pl.col("idx") <= i))
        ax1.plot(df_slice.select(pl.col("mean")).to_numpy(), linestyle="--", color='black', alpha=0.35, label=all_name)


        
        # ax0.plot(x, naive_dist, 'k--', label="Naive")

        ax0.plot(x, bary, label="Manual", color=cols[len(marginal_dists)], linewidth=2)
        ax0.set_title("Barycentre of Marginals")
        ax0.set_xlim(265, 305)
        ax0.set_ylim(0.0, 2.2)
        ax0.legend(loc="best")
        # ax0 = tidy_legend(ax0)
        ax0.set(xlabel="Temperature", ylabel="Density")

        ax1.plot(
            np.arange(len(bary_mus)) * step,
            bary_mus,
            color=cols[len(marginal_dists)],
            label="Ensemble",
        )
        ax1.fill_between(
            np.arange(len(bary_mus)) * step,
            bmu_array - bsig_array,
            bmu_array + bsig_array,
            color=cols[len(marginal_dists)],
            alpha=0.2,
        )
        ax1.set_ylim(270, 295)
        ax1.set_xlim(0, n_days)
        ax1.set(xlabel="Time index", ylabel="Temperature")
        ax1.legend(loc="best")
        # ax1 = tidy_legend(ax1)

        sns.despine()
        plt.savefig(f"output/barycentre_figs/monthly_{i+1:03d}.png")
        plt.close()

    images = []
    filenames = sorted(glob("output/barycentre_figs/monthly*.png"))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave("moving_barycentre_monthly.gif", images, format="GIF", fps=2)
    subprocess.run(
        [
            "gifsicle",
            "-i",
            "moving_barycentre_monthly.gif",
            "-O3",
            "--colors",
            "256",
            "-o",
            "moving_barycentre_opt_monthly.gif",
        ]
    )
