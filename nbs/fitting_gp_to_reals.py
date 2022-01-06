# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python [conda env:bayesian_ensembles]
#     language: python
#     name: conda-env-bayesian_ensembles-py
# ---

# +
import ensembles as es
import gpflow
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def report_on_percentiles(y_pred, y, y_std):

    n = len(y.ravel())

    n1 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 1)
    n2 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 2)
    n3 = np.sum(np.abs(y_pred.ravel() - y.ravel()) <= y_std.ravel() * 3)
    # print('Using {} data points'.format(n))
    print('{} within 1 std'.format(100 * n1 / n))
    print('{} within 2 std'.format(100 * n2 / n))
    print('{} within 3 std'.format(100 * n3 / n))

    return


# -

data_paths = glob('./data/*CESM*.csv')
df = pd.read_csv(data_paths[0], index_col='time')


def fit_GP(real_name):
    
    # 20 years worth of training data
    n_vals_train = 36500 // 5
    n_vals_test = 2500
    
    n_years = n_vals_train / 365    
    
    # Create kernel
    base_kernel = gpflow.kernels.Matern32()
    periodic_kernel = gpflow.kernels.Periodic(base_kernel)
    long_kernel = gpflow.kernels.SquaredExponential()
    short_kernel = gpflow.kernels.RationalQuadratic()
    noise_kernel = gpflow.kernels.White()
    # gpflow.kernels.ArcCosine

    kernel = periodic_kernel + long_kernel + short_kernel + noise_kernel
    kernel.kernels[0].period.assign(3.5 / n_years) # This is roughly 365 days
    kernel.kernels[1].lengthscales.assign(30 * 3.5 / n_years) # Roughly a climatological lengthscale 
    kernel.kernels[2].lengthscales.assign(0.5 * 3.5 / n_years) # Short, seasonal lengthscale
    
    # Collect, preprocess, and scale data
    x = np.arange(len(df)).reshape(-1,1).astype(np.float64)
    x_test = x[n_vals_train:n_vals_test + n_vals_train]
    x = x[:n_vals_train]

    y = df[[real_name]].values
    y_test = y[n_vals_train:n_vals_test + n_vals_train]
    y = y[:n_vals_train]
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_transformed = (y - y_mean) / y_std

    x_mean = np.mean(x)
    x_std = np.std(x)
    x_transformed = (x - x_mean) / x_std
    
    # Fit model
    models = es.SparseGP(kernel)
    params = es.SGPRParameters().to_dict()
    params['optim_nits'] = 2500
    params['n_inducing'] = 125
    
    init_method = robustgp.ConditionalVariance()
    Z = init_method.compute_initialisation(x_transformed, params['n_inducing'], kernel)[0]
    models.fit(x_transformed, y_transformed, Z, params)
    
    # Test
    x_test_transformed = (x_test - x_mean) / x_std
    y_test_transformed = (y_test - y_mean) / y_std
    mu, sigma2 = models.predict(x_test_transformed, params)
    mu = mu.numpy().squeeze()
    sigma2 = sigma2.numpy().squeeze()
    report_on_percentiles(mu, y_test_transformed, np.sqrt(sigma2))
    
    # Produce mu and std for time
    mu, sigma2 = models.predict(np.vstack([x_transformed, x_test_transformed]), params)
    report_on_percentiles(mu.numpy(), np.vstack([y_transformed, y_test_transformed]), np.sqrt(sigma2.numpy()))
    
    return np.vstack([x_transformed, x_test_transformed]), mu.numpy(), np.sqrt(sigma2)


# +
model_name = data_paths[0].split('/')[-1].split('_')[-1][:-4]

for real_name in df.columns:
    df_real = pd.DataFrame()
    x, mu, std = fit_GP(real_name)
    df_real['x'] = x.ravel()
    df_real['mu'] = mu.ravel()
    df_real['std'] = std.ravel()
    
    plt.plot(x, mu)
    
    df_real.to_csv(f'./data/{model_name}_{real_name}_fitted.csv')
# -


