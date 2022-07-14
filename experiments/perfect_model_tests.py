# Experiment script for performing perfect model tests.

import ensembles as es
from jax.config import config
import seaborn as sns
from glob import glob 
import numpy as np 
import xarray as xr
from ensembles.utils import PerfectModelTest


config.update("jax_enable_x64", True)
sns.set_style('whitegrid')


def get_data(ssp_dir):

    obs_da = xr.open_dataarray('./../experiments/data/obs/gmst/HadCRUT.5.0.1.0.analysis.anomalies_gmst.nc')
    obs_time = obs_da.time

    # Find the union between historical models and forecast models for this ssp
    hist_model_files = dict()
    for model_file in sorted(glob('./../experiments/data/gmst/historical/*nc')):
        model_name = '_'.join(model_file.split("/")[-1].split('_')[:2])
        hist_model_files[model_name] = model_file
    ssp_model_files = dict()
    for model_file in sorted(glob(ssp_dir + '/*nc')):
        model_name = '_'.join(model_file.split("/")[-1].split('_')[:2])
        ssp_model_files[model_name] = model_file
    model_overlap = np.intersect1d(list(hist_model_files.keys()), list(ssp_model_files.keys())).tolist()
    hist_model_files = {k:hist_model_files[k] for k in hist_model_files if k in model_overlap}
    ssp_model_files = {k:ssp_model_files[k] for k in ssp_model_files if k in model_overlap}

    # Load historical modes and calculate the anomally, and climatology (for the forecast models)
    hist_anom_models = []
    climatology_dict = dict()
    for mn, mf in hist_model_files.items():
        # Read in model data into a ProcessModel datatype
        da = xr.open_dataarray(mf)
        # Use observational time for ease of comparison between models and obs
        da['time'] = obs_time
        # Put data into a ProcessModel class
        model_data = es.ProcessModel(da, mn)
        # Find the anomally of that data
        anomaly_model = model_data.calculate_anomaly(resample_freq='Y')
        hist_anom_models.append(anomaly_model)
        climatology_dict[mn] = anomaly_model.climatology

    hindcast_models = es.ModelCollection(hist_anom_models)

    # Load forecast models and calculate the anomally
    ssp_anom_models = []
    for mn, mf in ssp_model_files.items():
        # Read in model data into a ProcessModel datatype
        da = xr.open_dataarray(mf)
        model_data = es.ProcessModel(da, mn)
        # Find the anomally of that data
        anomaly_model = model_data.calculate_anomaly(climatology=climatology_dict[mn], resample_freq='Y')
        ssp_anom_models.append(anomaly_model)

    forecast_models = es.ModelCollection(ssp_anom_models)
    return hindcast_models, forecast_models

for scenario in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']:
    hindcast_models, forecast_models = get_data(f'./../experiments/data/gmst/{scenario}')

    # Perform the perfect model test for MMM
    pmt = PerfectModelTest(
         hindcast_models=hindcast_models,
         forecast_models=forecast_models,
         emulate_method=es.MeanFieldApproximation,
         weight_method=es.UniformWeight,
         ensemble_method=es.MultiModelMean
         )

    pmt.run(n_optim_nits=2000, save_file=f'results/perfect_model_test_mmm_{scenario}.csv')

    # Perform the perfect model test for Barycentre
    pmt = PerfectModelTest(
            hindcast_models=hindcast_models,
            forecast_models=forecast_models,
            emulate_method=es.GPDTW1D,
            weight_method=es.LogLikelihoodWeight,
            ensemble_method=es.Barycentre
            )

    pmt.run(n_optim_nits=2000, save_file=f'results/perfect_model_test_barycentre_{scenario}.csv')

