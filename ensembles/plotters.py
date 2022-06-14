from fileinput import filename
from isort import file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typing as tp
import pandas as pd
from cycler import cycler


def cmap():
    return sns.color_palette("Set2", 8)


def get_style_cycler():
    style_cycler = 4 * cycler(color=sns.color_palette("Set2")) + cycler(
        ls=["-"] * 8 + ["--"] * 8 + ["-."] * 8 + [":"] * 8
    )
    return style_cycler


def _unique_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc="best")
    return ax


# TODO: add in barycentre plotting (and gif creation?)
def plot_distributions():

    return


# def plot_realisations(
#     X: np.ndarray, Y: np.ndarray, realisations: np.ndarray, ax, filename: str = None
# ):
#     [ax.plot(X, r, alpha=0.3, color="tab:blue", label="Realisation") for r in realisations.T]
#     ax.plot(X, Y, color="tab:orange", label="Truth")
#     ax = _unique_legend(ax)
#     sns.despine()
#     if filename:
#         plt.savefig(filename)
#     return ax


# def plot_individual_preds(
#     Xte: np.ndarray,
#     truth: np.ndarray,
#     individual_preds: tp.List[tp.Tuple[np.ndarray, np.ndarray]],
#     filename: str = None,
#     data_filename: str = None,
# ):
#     mus = np.vstack([i[0].numpy() for i in individual_preds])
#     sigmas = np.vstack([i[1].numpy() for i in individual_preds])

#     n_realisations = len(individual_preds)
#     n_obs = individual_preds[0][0].numpy().shape[0]
#     Xtes = np.tile(Xte.squeeze(), n_realisations).reshape(-1, 1)
#     Ylong = truth.reshape(-1, 1)
#     idx = np.tile(np.arange(n_realisations), n_obs).reshape(-1, 1)
#     # results = pd.DataFrame(
#     #     np.hstack((Xtes, Ylong, mus, sigmas, idx)), columns=["X", "Y", "mu", "sigma", "idx"]
#     # )
#     results = pd.DataFrame(np.hstack((Xtes, Ylong, mus, idx)), columns=["X", "Y", "mu", "idx"])
#     results = results.melt(id_vars=["X", "idx"])
#     g = sns.FacetGrid(results, col="idx", col_wrap=np.minimum(n_realisations, 3), despine=True)
#     g.map_dataframe(sns.lineplot, x="X", y="value", hue="variable")
#     g.add_legend()
#     if filename:
#         plt.savefig(filename)
#     return g


# def plot_group_pred(
#     mu, sigma, Xtr, realisations, latent_y, Xte, ax, n_stds: int = 3, filename: str = None
# ):
#     std_alpha = 0.1
#     for j in range(1, n_stds + 1):
#         ax.fill_between(Xte, mu - j * sigma, mu + j * sigma, alpha=std_alpha * j, color="tab:blue")
#     ax.plot(Xte, mu, color="tab:blue", label="Pred")
#     ax.plot(Xtr, latent_y, color="tab:orange", label="Truth")
#     [
#         ax.plot(Xtr, r, alpha=0.3, color="tab:orange", label="Realisation", linestyle="--")
#         for r in realisations.T
#     ]
#     ax = _unique_legend(ax)
#     sns.despine()
#     if filename:
#         plt.savefig(filename)
#     return ax

# TODO: add in barycentre plotting (and gif creation?)
