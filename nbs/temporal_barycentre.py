import polars as pl
import ot
import matplotlib.pyplot as plt
import numpy as np
import imageio
from glob import glob
import seaborn as sns
import subprocess
from tqdm import trange, tqdm


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


if __name__ == "__main__":
    n_bins = 500
    n_days = 732
    step = 3
    x = np.arange(n_bins, dtype=np.float64)
    build_gaussian = lambda v: ot.datasets.make_1D_gauss(n_bins, m=v[0], s=v[1])

    df_read = pl.concat([pl.read_csv(m) for m in glob("output/preds/*.csv")])
    standardisers = pl.read_csv("output/aux/standardisers.csv")
    df_read = df_read.join(standardisers, on="model", how="left")

    df_read = df_read.with_columns(
        ((pl.col("pred_mu") * pl.col("sd_mean")) + pl.col("mean_mean")).alias("mean")
    )
    df_read = df_read.with_columns(
        ((pl.col("pred_sigma") * pl.col("sd_sd")) + pl.col("mean_sd")).alias("sd")
    )
    n_obs = (
        df_read.groupby("model")
        .agg([pl.col("mean").count()])
        .select(pl.col("mean"))
        .to_numpy()[0][0]
    )
    bary_mus = []
    bary_sigmas = []

    for i in tqdm(range(0, n_days, step)):
        df = df_read.melt(id_vars=["idx", "model"], value_vars=["mean", "sd"])
        df = df.filter(pl.col("idx") == i)
        df = df.pivot(values="value", index="model", columns="variable")
        marginals = df.select(["mean", "sd"]).to_numpy()
        marginal_dists = [build_gaussian(m) for m in marginals]
        marginal_names = [n for n in df.to_series(0)]

        n_distributions = marginals.shape[0]
        means = marginals[:, 0]
        std_devs = marginals[:, 1]
        weights = np.array([1 / n_distributions] * n_distributions)
        weights2 = np.array([0.25] * 4)
        bary = gaussian_barycentre(means, std_devs, weights, as_hist=True, n_bins=500)
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

        ax0.plot(x, bary, label="Manual", color=cols[len(marginal_dists)], linewidth=2)
        ax0.set_title("Barycentre of Marginals")
        ax0.set_xlim(265, 305)
        ax0.set_ylim(0.0, 0.16)
        ax0.legend(loc="best")
        ax0 = tidy_legend(ax0)
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
        ax1 = tidy_legend(ax1)

        sns.despine()
        plt.savefig(f"output/barycentre_figs/{i+1:03d}.png")
        plt.close()

    images = []
    filenames = sorted(glob("output/barycentre_figs/*.png"))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave("moving_barycentre.gif", images, format="GIF", fps=10)
    subprocess.run(
        [
            "gifsicle",
            "-i",
            "moving_barycentre.gif",
            "-O3",
            "--colors",
            "256",
            "-o",
            "moving_barycentre_opt.gif",
        ]
    )
