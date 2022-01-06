# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python [conda env:bayesian_ensembles]
#     language: python
#     name: conda-env-bayesian_ensembles-py
# ---

import ensembles as es
import gpflow
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

data_paths = glob('/home/jupyter/bayesian_ensembling/nbs/data/*.csv')
data_one = pd.read_csv(data_paths[0], index_col='time')
data_one.head()

x = np.arange(y.shape[0]).reshape(-1,1).astype(np.float64)
x_test = x[40000:]
x = x[:40000]

y = data_one[['r10i1p1']].values
y_test = y[40000:]
y = y[:40000]

plt.plot(x, y)

# +
y_mean = np.mean(y)
y_std = np.std(y)

y_transformed = (y - y_mean) / y_std

x_mean = np.mean(x)
x_std = np.std(x)

x_transformed = (x - x_mean) / x_std
# -

plt.hist(y_transformed, bins=100)
plt.show()

base_kernel = gpflow.kernels.Matern32()
kernel = gpflow.kernels.Periodic(base_kernel) + gpflow.kernels.Linear()
# model = es.SparseGP

# Prior sampling

def prior_sample(x, kernel):
    mu = np.zeros(x.shape[0])
    sigma = kernel.K(x, x)
    
    return np.random.multivariate_normal(mu, sigma, 10)


x_query = np.linspace(-5, 5, 200).reshape(-1, 1)
y_query = prior_sample(x_query, base_kernel)

y_periodic_query = prior_sample(x_query, kernel)

plt.plot(x_query, y_query.T)
plt.show()

plt.plot(x_query, y_periodic_query.T)
plt.show()

kernel.kernels[0].period.assign(0.1)

# Build models

models = es.SparseGP(kernel)

gpflow.utilities.print_summary(models.model)

params = es.SGPRParameters().to_dict()

models.fit(x_transformed, y_transformed, params)

x_test_transformed = (x_test - x_mean) / x_std
y_test_transformed = (y_test - y_mean) / y_std

mu, sigma2 = models.predict(x_test_transformed, params)
mu = mu.numpy().squeeze()
sigma2 = sigma2.numpy().squeeze()

# +
fig, ax = plt.subplots(figsize=(12, 6))

# [ax.axvline(x = i, color='black', alpha=0.2, linestyle='--', linewidth=1, label='Inducing location') for i in models.Z.Z.numpy().tolist()]
ax.plot(x_test_transformed, mu, color='tab:orange', label='Predictive mean')
ax.fill_between(x_test_transformed.ravel(), mu - np.sqrt(sigma2), mu + np.sqrt(sigma2), alpha=0.3, color='tab:orange', label=r'$1\sigma$')
ax.plot(x_test_transformed, y_test_transformed, color='tab:blue')
# ax.plot(x, y, '+', color='tab:blue', alpha=0.02, label='Observations')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(xlabel=r'x', ylabel=r'f(x)', title=f'{models.name} model predictions')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='best')

plt.xlim([2.2, 2.4])
# -

mu, sigma2 = models.predict(x_transformed, params)
mu = mu.numpy().squeeze()
sigma2 = sigma2.numpy().squeeze()

# +
fig, ax = plt.subplots(figsize=(12, 6))

# [ax.axvline(x = i, color='black', alpha=0.2, linestyle='--', linewidth=1, label='Inducing location') for i in models.Z.Z.numpy().tolist()]
ax.plot(x_transformed, mu, color='tab:orange', label='Predictive mean')
ax.fill_between(x_transformed.ravel(), mu - np.sqrt(sigma2), mu + np.sqrt(sigma2), alpha=0.3, color='tab:orange', label=r'$1\sigma$')
ax.plot(x_transformed, y_transformed, color='tab:blue')
# ax.plot(x, y, '+', color='tab:blue', alpha=0.02, label='Observations')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(xlabel=r'x', ylabel=r'f(x)', title=f'{models.name} model predictions')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='best')

plt.xlim([0, 0.2])
# -

models.model


