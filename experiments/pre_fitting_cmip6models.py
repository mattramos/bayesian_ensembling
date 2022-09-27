# I think the paths in this will be wrong

import ensembles as es
import matplotlib.pyplot as plt
import jax.random as jr
from jax.config import config
import seaborn as sns
from glob import glob 
import numpy as np 
import xarray as xr
from ensembles.plotters import _unique_legend
import pickle
import os

config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)
sns.set_style('whitegrid')


def load_model_data(ssp_dir='./data/gmst/ssp370'):

    obs_da = xr.open_dataarray('./data/obs/gmst/HadCRUT.5.0.1.0.analysis.anomalies_gmst.nc')
    obs_time = obs_da.time

    # Find the union between historical models and forecast models for this ssp
    hist_model_files = dict()
    for model_file in sorted(glob('./data/gmst/historical/*nc')):
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

    hist_anom_models = es.ModelCollection(hist_anom_models)

    # Load forecast models and calculate the anomally
    ssp_anom_models = []
    for mn, mf in ssp_model_files.items():
        # Read in model data into a ProcessModel datatype
        da = xr.open_dataarray(mf)
        model_data = es.ProcessModel(da, mn)
        # Find the anomally of that data
        # time = da.indexes['time']
        # if not isinstance(time, DatetimeIndex):
        #     datetimeindex = da.indexes['time'].to_datetimeindex()
        #     da['time'] = datetimeindex
        anomaly_model = model_data.calculate_anomaly(climatology=climatology_dict[mn], resample_freq='Y')
        ssp_anom_models.append(anomaly_model)

    ssp_anom_models = es.ModelCollection(ssp_anom_models)

    return hist_anom_models, ssp_anom_models

def save_models(ssp_dir='./data/gmst/ssp370'):
    ssp_num = ssp_dir.split('/')[-1]
    hist_anom_models, ssp_anom_models = load_model_data(ssp_dir=ssp_dir)
    hist_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2500, progress_bar=True)
    ssp_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2500, progress_bar=True)

    if not os.path.exists('./pre_fit_models'):
        os.makedirs('./pre_fit_models')

    hist_save_path = './pre_fit_models/hist{}_1D_models.pkl'.format(ssp_num)
    with open(hist_save_path, 'wb') as file:
        pickle.dump(hist_anom_models, file) 
    ssp_save_path = './pre_fit_models/{}_1D_models.pkl'.format(ssp_num)
    with open(ssp_save_path, 'wb') as file:
        pickle.dump(ssp_anom_models, file)
        pickle.dump(ssp_anom_models, file)


ssps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp460', 'ssp434', 'ssp585']
for ssp in ssps:
    ssp_dir = './data/gmst/{}'.format(ssp)
    save_models(ssp_dir=ssp_dir)