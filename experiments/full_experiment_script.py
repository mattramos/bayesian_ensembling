# The main experiment file for GMST
# Script which runs all the scenarios, plots a result and retains some key values.

import ensembles as es
import matplotlib.pyplot as plt
import jax.random as jr
from jax.config import config
import seaborn as sns
from glob import glob 
import numpy as np 
import xarray as xr
from ensembles.plotters import _unique_legend

config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)
sns.set_style('whitegrid')


# Load obs
obs_da = xr.open_dataarray('data/obs/gmst/HadCRUT.5.0.1.0.analysis.anomalies_gmst.nc')
obs_time = obs_da.time
obs_da = obs_da.resample(time='Y').mean()
observations = es.ProcessModel(obs_da, model_name='Observations')
observations.plot()
sns.despine()

# Load models
def load_model_data(ssp_dir='data/gmst/ssp370'):

    obs_da = xr.open_dataarray('data/obs/gmst/HadCRUT.5.0.1.0.analysis.anomalies_gmst.nc')
    obs_time = obs_da.time

    # Find the union between historical models and forecast models for this ssp
    hist_model_files = dict()
    for model_file in sorted(glob('data/gmst/historical/*nc')):
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
        anomaly_model = model_data.calculate_anomaly(climatology=climatology_dict[mn], resample_freq='Y')
        ssp_anom_models.append(anomaly_model)

    ssp_anom_models = es.ModelCollection(ssp_anom_models)

    return hist_anom_models, ssp_anom_models

hist119_anom_models, ssp119_anom_models = load_model_data(ssp_dir='data/gmst/ssp119')
hist126_anom_models, ssp126_anom_models = load_model_data(ssp_dir='data/gmst/ssp126')
hist245_anom_models, ssp245_anom_models = load_model_data(ssp_dir='data/gmst/ssp245')
hist370_anom_models, ssp370_anom_models = load_model_data(ssp_dir='data/gmst/ssp370')
hist434_anom_models, ssp434_anom_models = load_model_data(ssp_dir='data/gmst/ssp434')
hist460_anom_models, ssp460_anom_models = load_model_data(ssp_dir='data/gmst/ssp460')
hist585_anom_models, ssp585_anom_models = load_model_data(ssp_dir='data/gmst/ssp585')

# Construct model posteriors

hist119_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp119_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

hist126_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp126_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

hist245_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp245_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

hist370_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp370_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

hist434_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp434_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

hist460_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp460_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

hist585_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

ssp585_anom_models.fit(model=es.GPDTW1D(), compile_objective=True, n_optim_nits=2000, progress_bar=False)

# Construct weights for each model
weight_function = es.CRPSWeight()
hist119_cprs_weights = weight_function(hist119_anom_models, observations)
hist126_cprs_weights = weight_function(hist126_anom_models, observations)
hist245_cprs_weights = weight_function(hist245_anom_models, observations)
hist370_cprs_weights = weight_function(hist370_anom_models, observations)
hist434_cprs_weights = weight_function(hist434_anom_models, observations)
hist460_cprs_weights = weight_function(hist460_anom_models, observations)
hist585_cprs_weights = weight_function(hist585_anom_models, observations)

# Flatten weights
# 1 weight per model
weights_119 = hist119_cprs_weights.mean('time').expand_dims(time=ssp119_anom_models[0].model_data.time, axis=1)
weights_126 = hist126_cprs_weights.mean('time').expand_dims(time=ssp126_anom_models[0].model_data.time, axis=1)
weights_245 = hist245_cprs_weights.mean('time').expand_dims(time=ssp245_anom_models[0].model_data.time, axis=1)
weights_370 = hist370_cprs_weights.mean('time').expand_dims(time=ssp370_anom_models[0].model_data.time, axis=1)
weights_434 = hist434_cprs_weights.mean('time').expand_dims(time=ssp434_anom_models[0].model_data.time, axis=1)
weights_460 = hist460_cprs_weights.mean('time').expand_dims(time=ssp460_anom_models[0].model_data.time, axis=1)
weights_585 = hist585_cprs_weights.mean('time').expand_dims(time=ssp585_anom_models[0].model_data.time, axis=1)

# Construct the barycentres
ensemble_method = es.Barycentre()
ssp119_barycentre = ensemble_method(ssp119_anom_models, weights_119)
ssp126_barycentre = ensemble_method(ssp126_anom_models, weights_126)
ssp245_barycentre = ensemble_method(ssp245_anom_models, weights_245)
ssp370_barycentre = ensemble_method(ssp370_anom_models, weights_370)
ssp434_barycentre = ensemble_method(ssp434_anom_models, weights_434)
ssp460_barycentre = ensemble_method(ssp460_anom_models, weights_460)
ssp585_barycentre = ensemble_method(ssp585_anom_models, weights_585)

# Save values of temp at 2050 - 1010
upper = round((ssp119_barycentre.mean + 2 * np.sqrt(ssp119_barycentre.variance)).values[35], 3)
mean = round(ssp119_barycentre.mean.values[35], 3)
lower = round((ssp119_barycentre.mean - 2 * np.sqrt(ssp119_barycentre.variance)).values[35], 3)
print(f'ssp119 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp119_barycentre.mean + 2 * np.sqrt(ssp119_barycentre.variance)).values[85], 3)
mean = round(ssp119_barycentre.mean.values[85], 3)
lower = round((ssp119_barycentre.mean - 2 * np.sqrt(ssp119_barycentre.variance)).values[85], 3)
print(f'ssp119 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp126_barycentre.mean + 2 * np.sqrt(ssp126_barycentre.variance)).values[35], 3)
mean = round(ssp126_barycentre.mean.values[35], 3)
lower = round((ssp126_barycentre.mean - 2 * np.sqrt(ssp126_barycentre.variance)).values[35], 3)
print(f'ssp126 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp126_barycentre.mean + 2 * np.sqrt(ssp126_barycentre.variance)).values[85], 3)
mean = round(ssp126_barycentre.mean.values[85], 3)
lower = round((ssp126_barycentre.mean - 2 * np.sqrt(ssp126_barycentre.variance)).values[85], 3)
print(f'ssp126 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp245_barycentre.mean + 2 * np.sqrt(ssp245_barycentre.variance)).values[35], 3)
mean = round(ssp245_barycentre.mean.values[35], 3)
lower = round((ssp245_barycentre.mean - 2 * np.sqrt(ssp245_barycentre.variance)).values[35], 3)
print(f'ssp245 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp245_barycentre.mean + 2 * np.sqrt(ssp245_barycentre.variance)).values[85], 3)
mean = round(ssp245_barycentre.mean.values[85], 3)
lower = round((ssp245_barycentre.mean - 2 * np.sqrt(ssp245_barycentre.variance)).values[85], 3)
print(f'ssp245 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp370_barycentre.mean + 2 * np.sqrt(ssp370_barycentre.variance)).values[35], 3)
mean = round(ssp370_barycentre.mean.values[35], 3)
lower = round((ssp370_barycentre.mean - 2 * np.sqrt(ssp370_barycentre.variance)).values[35], 3)
print(f'ssp370 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp370_barycentre.mean + 2 * np.sqrt(ssp370_barycentre.variance)).values[85], 3)
mean = round(ssp370_barycentre.mean.values[85], 3)
lower = round((ssp370_barycentre.mean - 2 * np.sqrt(ssp370_barycentre.variance)).values[85], 3)
print(f'ssp370 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp434_barycentre.mean + 2 * np.sqrt(ssp434_barycentre.variance)).values[35], 3)
mean = round(ssp434_barycentre.mean.values[35], 3)
lower = round((ssp434_barycentre.mean - 2 * np.sqrt(ssp434_barycentre.variance)).values[35], 3)
print(f'ssp434 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp434_barycentre.mean + 2 * np.sqrt(ssp434_barycentre.variance)).values[85], 3)
mean = round(ssp434_barycentre.mean.values[85], 3)
lower = round((ssp434_barycentre.mean - 2 * np.sqrt(ssp434_barycentre.variance)).values[85], 3)
print(f'ssp434 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp460_barycentre.mean + 2 * np.sqrt(ssp460_barycentre.variance)).values[35], 3)
mean = round(ssp460_barycentre.mean.values[35], 3)
lower = round((ssp460_barycentre.mean - 2 * np.sqrt(ssp460_barycentre.variance)).values[35], 3)
print(f'ssp460 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp460_barycentre.mean + 2 * np.sqrt(ssp460_barycentre.variance)).values[85], 3)
mean = round(ssp460_barycentre.mean.values[85], 3)
lower = round((ssp460_barycentre.mean - 2 * np.sqrt(ssp460_barycentre.variance)).values[85], 3)
print(f'ssp460 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp585_barycentre.mean + 2 * np.sqrt(ssp585_barycentre.variance)).values[35], 3)
mean = round(ssp585_barycentre.mean.values[35], 3)
lower = round((ssp585_barycentre.mean - 2 * np.sqrt(ssp585_barycentre.variance)).values[35], 3)
print(f'ssp585 mean at 2050: {mean} ({lower}-{upper}) 95% credible interval')

upper = round((ssp585_barycentre.mean + 2 * np.sqrt(ssp585_barycentre.variance)).values[85], 3)
mean = round(ssp585_barycentre.mean.values[85], 3)
lower = round((ssp585_barycentre.mean - 2 * np.sqrt(ssp585_barycentre.variance)).values[85], 3)
print(f'ssp585 mean at 2100: {mean} ({lower}-{upper}) 95% credible interval')


# Plotting
from ensembles.plotters import cmap

from scipy import stats
def plot_dist(dist, color='tab:blue', label='None', alpha=0.2, order=3):
    plt.plot(dist.mean.time, dist.mean, color=color, label=label, zorder=order)
    plt.fill_between(dist.mean.time.values, dist.mean - 2 * np.sqrt(dist.variance), dist.mean + 2 * np.sqrt(dist.variance), alpha=alpha, color=color, zorder=order-1, linewidth=0)

    return    
plt.rcParams['pdf.fonttype'] = 'truetype'

plt.figure(figsize=(6.5, 4))
plot_dist(ssp119_barycentre, color=cmap()[0], label='ssp119')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp119_with_95percent_credible_interval.pdf")
plt.show()


plt.figure(figsize=(6.5, 4))
plot_dist(ssp126_barycentre, color=cmap()[0], label='ssp126')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp126_with_95percent_credible_interval.pdf")
plt.show()

plt.figure(figsize=(6.5, 4))
plot_dist(ssp245_barycentre, color=cmap()[0], label='ssp245')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp245_with_95percent_credible_interval.pdf")
plt.show()


plt.figure(figsize=(6.5, 4))
plot_dist(ssp370_barycentre, color=cmap()[0], label='ssp370')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp370_with_95percent_credible_interval.pdf")
plt.show()


plt.figure(figsize=(6.5, 4))
plot_dist(ssp434_barycentre, color=cmap()[0], label='ssp434')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp434_with_95percent_credible_interval.pdf")
plt.show()


plt.figure(figsize=(6.5, 4))
plot_dist(ssp460_barycentre, color=cmap()[0], label='ssp460')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp460_with_95percent_credible_interval.pdf")
plt.show()

plt.figure(figsize=(6.5, 4))
plot_dist(ssp585_barycentre, color=cmap()[0], label='ssp585')
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/ssp585_with_95percent_credible_interval.pdf")
plt.show()

plt.figure(figsize=(6.5, 4))
labels = ['ssp245', 'ssp370', 'ssp585']
for i, forecast in enumerate([ssp245_barycentre, ssp370_barycentre, ssp585_barycentre]):
    plot_dist(forecast, color=cmap()[i], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
plt.legend()
sns.despine()
plt.savefig("figures/figure1_with_95percent_credible_interval.pdf")
plt.show()