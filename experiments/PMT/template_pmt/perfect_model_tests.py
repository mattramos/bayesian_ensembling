# Experiment script for performing perfect model tests.

import ensembles as es
from jax.config import config
import seaborn as sns
from glob import glob 
import numpy as np 
import xarray as xr
from ensembles.utils import PerfectModelTest
import pickle as pkl


config.update("jax_enable_x64", True)
sns.set_style('whitegrid')

# Load observations
obs_da = xr.open_dataarray('./../../data/obs/gmst/HadCRUT.5.0.1.0.analysis.anomalies_gmst.nc')
obs_time = obs_da.time

def load_prefit_models(ssp):
    hist_file = './../../pre_fit_models/hist{}_1D_models.pkl'.format(ssp)
    ssp_file = './../../pre_fit_models/ssp{}_1D_models.pkl'.format(ssp)
    with open(hist_file, 'rb') as file1:
        hist_models = pkl.load(file1)
    with open(ssp_file, 'rb') as file2:
        ssp_models = pkl.load(file2)
    return hist_models, ssp_models


for scenario in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']:
    for weight_method in [es.weights.KSDWeight, es.weights.LogLikelihoodWeight, es.weights.InverseSquareWeight, es.weights.CRPSWeight, es.weights.UniformWeight]:
        print()
        print('Running perfect model test for {} with {} weights'.format(scenario, weight_method().name))
        print()
        hindcast_models, forecast_models = load_prefit_models(scenario)

        # Perform the perfect model test for Barycentre
        pmt = PerfectModelTest(
                hindcast_models=hindcast_models,
                forecast_models=forecast_models,
                emulate_method=es.GPDTW1D,
                weight_method=weight_method,
                ensemble_method=es.Barycentre,
                ssp=scenario,
                save_fig_dir='./results/'
                )

        pmt.run(n_optim_nits=1000, use_prefit_models=True)

