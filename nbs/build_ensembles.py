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
#     display_name: 'Python 3.8.12 64-bit (''bayesian_ensembles'': conda)'
#     language: python
#     name: python3
# ---

# %%
import polars as pl
import ensembles as es
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# %%
data_paths = glob('/home/pindert2/bayesian_ensembling/nbs/data/*.csv')
data_one = pl.read_csv(data_paths[0])
data_one.head()

# %%
data_paths

# %%
value_matrix = data_one[:, 1:].to_pandas().values.astype(np.float64)

fig, ax = plt.subplots(figsize = (30, 6))

for v in value_matrix.T:
    ax.plot(np.arange(data_one['time'].shape[0]), v, color ='tab:blue', alpha=0.2)

ax.set(xlabel = 'Time index', ylabel = 'Temperature')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# %%
value_matrix.shape

# %%
mus = np.mean(value_matrix, axis=1)
sigma = np.sqrt(np.var(value_matrix, axis=1))
idx = np.arange(mus.shape[0])


fig, ax = plt.subplots(figsize=(30, 6))
for v in value_matrix.T:
    ax.plot(idx, v, color ='tab:blue', alpha=0.1)

# ax.plot(idx, mus, color='tab:orange')
# ax.fill_between(idx, mus - sigma, mus + sigma, alpha=0.3, color='tab:orange', label=r'$1\sigma$')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 1000)
ax.set(xlabel = 'Time index', ylabel='Temperature')
plt.show()

# %% [markdown]
#
#
#
# 1. One model per realisation
#     * Aggregate models
# 2. Define a model over the data distribution

# %% [markdown]
#
