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

import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# +
models = ['CESM',
          'ECEARTH',
          'CANESM',
          'CSIRO',
          'GFDL-esm2m'
         ]

# Could replace with cftime or netcdf-time to get a meaningful time axis
time = np.arange(9800)

colors = plt.cm.tab10


# +
def load_fitted_reals():
    all_models = dict()
    
    for model in models:
        model_dict = dict()
        model_df = pd.read_csv(f'./data/SingLoc_{model}.csv', index_col='time')
        gp_preds = dict()
        for real in model_df.columns:
            real_mean = np.mean(model_df[real][:36500 // 5])
            real_std = np.std(model_df[real][:36500 // 5])
            try:
                df_real = pd.read_csv(f'./data/{model}_{real}_fitted.csv')
            except FileNotFoundError:
                break
            mu = df_real['mu'].values * real_std + real_mean
            sigma = df_real['std'].values * real_std
            gp_preds[real] = {
                'mu': mu,
                'sigma': sigma}
            
        model_dict['gp_preds'] = gp_preds
        model_dict['mean'] = np.mean(model_df, axis=1).values[:9800]
        model_dict['std'] = np.std(model_df, axis=1).values[:9800]
        all_models[model] = model_dict
    
    return all_models

def plot_real_marginals(models_dict):
    
    plt.figure(figsize=(12,10))
    for i, model in enumerate(models):
        for j, real in enumerate(models_dict[model]['gp_preds'].keys()):
            real_vals = models_dict[model]['gp_preds'][real]
            plt.fill_between(time,
                             real_vals['mu'] - real_vals['sigma'],
                             real_vals['mu'] + real_vals['sigma'],
                             alpha=0.01,
                             color=colors(i),
                             label=model if j == 0 else "",
                             zorder=1)
        plt.plot(time,
                 models_dict[model]['mean'],
                 color=colors(i),
                 label=model + ' mean',
                 zorder=3)
            
    plt.legend()
    plt.ylabel('Temperature (K)')
    plt.xlim([9000,9800])
    plt.show()
    
    return    

def plot_real_means(models_dict):
    
    plt.figure(figsize=(12,10))
    for i, model in enumerate(models):
        for j, real in enumerate(models_dict[model]['gp_preds'].keys()):
            real_vals = models_dict[model]['gp_preds'][real]
            plt.plot(time,
                     real_vals['mu'],
                     alpha=0.25,
                     color=colors(i),
                     label=model + ' realisations' if j == 0 else "",
                     zorder=1)
        plt.plot(time,
                 models_dict[model]['mean'],
                 color=colors(i),
                 label=model + ' mean',
                 zorder=3)
            
    plt.legend()
    plt.ylabel('Temperature (K)')
    plt.xlim([9000,9800])
    plt.show()
    
    return   
def compare_MMM(models_dict):
    plt.figure(figsize=(12,10))
    for i, model in enumerate(models):
        plt.subplot(3, 2, i + 1)
        plt.plot(time,
                 models_dict[model]['mean'] - models_dict[model]['std'],
                 color=colors(i),
                 ls='--',
                 lw=1,
                 zorder=3)
        plt.plot(time,
                 models_dict[model]['mean'] + models_dict[model]['std'],
                 color=colors(i),
                 ls='--',
                 lw=1,
                 label=model + ' std',
                 zorder=3)
        plt.plot(time,
                 models_dict[model]['mean'],
                 color=colors(i),
                 lw=1,
                 label=model + ' MMM',
                 zorder=3)
        for j, real in enumerate(models_dict[model]['gp_preds'].keys()):
            real_vals = models_dict[model]['gp_preds'][real]
            plt.fill_between(time,
                             real_vals['mu'] - real_vals['sigma'],
                             real_vals['mu'] + real_vals['sigma'],
                             alpha=0.01,
                             color=colors(i),
                             label=model + ' marginal' if j == 0 else "",
                             zorder=1)
        plt.legend()
        plt.ylabel('Temperature (K)')
        plt.xlim([9000,9800])
    plt.show()
    
    return

def plot_single_slice(models_dict, time_loc):
    plt.figure(figsize=(12,10))
    for i, model in enumerate(models):
        # plt.subplot(3, 2, i + 1)
        for j, real in enumerate(models_dict[model]['gp_preds'].keys()):
            real_vals = models_dict[model]['gp_preds'][real]
            real_time_mu = real_vals['mu'][time_loc]
            real_time_std = real_vals['sigma'][time_loc]
            x_ = np.linspace(272, 300, 200)
            plt.plot(x_,
                     norm.pdf(x_, real_time_mu, real_time_std),
                     color=colors(i),
                     label=model + ' marginal' if j == 0 else "",
                     zorder=1)
    plt.xlabel('Temperature (K)')
    plt.title('PDFs of marginals')
    plt.legend()
    plt.show()
    return


# -

models_dict = load_fitted_reals()
plot_real_marginals(models_dict)
plot_real_means(models_dict)
compare_MMM(models_dict)
plot_single_slice(models_dict, 9280)






