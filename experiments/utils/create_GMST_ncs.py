# Script used to turn gridded data into GMSTs and to collate across model realisations

from glob import glob
import numpy as np
import xarray as xr
import os
from pandas import DatetimeIndex


xr.set_options(keep_attrs=True)


def prepro(ds):
    # Add realisation variable
    model_real = ds.encoding['source'].split('_')[-1][:-3]
    ds = ds.assign_coords(realisation=model_real)

    # Do area weighted average
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"
    weighted_mean = ds.weighted(weights).mean(["longitude", "latitude"])

    if 'height' in weighted_mean.variables:
        weighted_mean = weighted_mean.drop('height')    
    return weighted_mean


# Preprocess observations
obs_files = glob('/home/amosm1/bayesian_ensembling/experiments/data/obs/gridded/*.nc')
obs_ds = xr.open_mfdataset(
    obs_files,
    combine='nested',
    concat_dim='realization').tas.load().sel(time=slice('1850-01-01', '2014-12-31'))
obs_ds = obs_ds.rename(realization='realisation')
time_coord = obs_ds.time # Keeping this to make models uniform
weights = np.cos(np.deg2rad(obs_ds.latitude))
weights.name = "weights"
weighted_mean = obs_ds.weighted(weights).mean(["longitude", "latitude"])
weighted_mean.to_netcdf('/home/amosm1/bayesian_ensembling/experiments/data/obs/gmst/HadCRUT.5.0.1.0.analysis.anomalies_gmst.nc')

# Preprocess models
scenario_dirs = glob('./../data/gridded/*')
for scenario_dir in scenario_dirs:
    # Check file exists
    if not os.path.exists(scenario_dir.replace('gridded', 'gmst')):
        os.mkdir(scenario_dir.replace('gridded', 'gmst'))

    # Find model files
    model_files = glob(scenario_dir + '/*.nc')
    unique_models = np.unique(['_'.join(model_file.split('/')[-1].split('_')[:3]) for model_file in model_files])
    for model in unique_models:
        files = [model_file for model_file in model_files if model == '_'.join(model_file.split('/')[-1].split('_')[:3])]

        # Load data
        ds = xr.open_mfdataset(
            files,
            preprocess=prepro,
            combine='nested',
            concat_dim='realisation').tas.load()

        # Change time coord to something generic
        time = ds.indexes['time']
        if not isinstance(time, DatetimeIndex):
            datetimeindex = ds.indexes['time'].to_datetimeindex()
            ds['time'] = datetimeindex
        print(f'Model: {model}, n_reals: {len(ds.realisation.values)}')
        # Save data

        # If CanESM5, split up the physics realisations
        if '_'.join(model.split('_')[:2]) == 'CCCma_CanESM5':
            model_name_p1 = model.replace('CanESM5', 'CanESM5p1')
            model_name_p2 = model.replace('CanESM5', 'CanESM5p2')

            try:
                ds_p1 = ds.sel(realisation=[real for real in ds.realisation.values if 'p1' in real])
            except KeyError:
                save_p1 == False
            try:
                ds_p2 = ds.sel(realisation=[real for real in ds.realisation.values if 'p2' in real])
            except KeyError:
                save_p2 == False
            if len(ds_p1.realisation) != 0:
                save_name = os.path.join(
                    scenario_dir.replace('gridded', 'gmst'),
                    model.replace('ESM5', 'ESM5-p1') + '_gmst.nc'
                    )
                ds_p1.to_netcdf(save_name)
            if len(ds_p2.realisation) != 0:
                save_name = os.path.join(
                    scenario_dir.replace('gridded', 'gmst'),
                    model.replace('ESM5', 'ESM5-p2') + '_gmst.nc'
                    )
                ds_p2.to_netcdf(save_name)


        else:
            save_name = os.path.join(
                scenario_dir.replace('gridded', 'gmst'),
                '_'.join(files[0].split('/')[-1].split('_')[:3]) + '_gmst.nc'
                )
            ds.to_netcdf(save_name)

