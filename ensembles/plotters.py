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