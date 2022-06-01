from concurrent.futures import process
from configparser import NoOptionError
from jax import jit
import typing as tp
import jax.numpy as jnp
from .data import ModelCollection, ProcessModel, Distribution
from .wasserstein import gaussian_barycentre
import abc
from tqdm import trange, tqdm
import numpy as np
import ot
import distrax
import matplotlib.pyplot as plt
import xarray as xr
import distrax as dx
from joblib import Parallel, delayed


class AbstractEnsembleScheme:
    def __init__(self, name: str) -> None:
        self.name = name
        self.distributions = None

    @abc.abstractmethod
    # TODO define a union array type of jnp and np arrays
    def _compute(
        self, process_models: ModelCollection, weights: xr.DataArray
    ) -> Distribution:
        raise NotImplementedError

    def __call__(
        self, process_models: ModelCollection, weights: xr.DataArray, **kwargs
    ) -> tp.Any:
        return self._compute(process_models=process_models, weights=weights, **kwargs)

# Commented out 30-5-22 MA in favour of parallel scheme below
# class Barycentre(AbstractEnsembleScheme):
#     def __init__(self, name: str = "Barycentre") -> None:
#         super().__init__(name)

#     @abc.abstractmethod
#     def _compute(self, process_models: ModelCollection, weights: xr.DataArray) -> Distribution:
#         # if process_models[0].model_data.ndim > 2:
#         #      raise NotImplementedError('Not implemented for more than temporal dimensions')
    
#         bary_mus = []
#         bary_stds = []
#         n_points = int(process_models[0].model_data.size / process_models[0].model_data.realisation.size)
#         weights = weights.values.reshape(process_models.number_of_models, -1)
#         # TODO want to parallelise this!
#         for t_idx in trange(n_points):
#             means_t = []
#             stds_t = []
#             for process_model in process_models:
#                 if not process_model.distribution:
#                     raise AttributeError(
#                         f"No posterior for model {t_idx}. Please run model.fit() first."
#                     )
#                 # Load pre-flattened distribution
#                 dist = process_model.distribution._dist
#                 mean = dist.mean()[t_idx]
#                 std = jnp.sqrt(dist.variance()[t_idx])
#                 means_t.append(mean)
#                 stds_t.append(std)
#             weight = weights[:, t_idx]
#             bary_mu, bary_std = gaussian_barycentre(np.asarray(means_t), np.asarray(stds_t), weight)

#             bary_mus.append(bary_mu)
#             bary_stds.append(bary_std)
#         blank_array = xr.ones_like(process_model.model_data[0].drop('realisation')) * np.nan
#         blank_array = blank_array.rename('blank')
#         ensemble_dist = Distribution(
#             mu=np.asarray(bary_mus), covariance=np.asarray(bary_stds) ** 2,
#             dim_array=blank_array,
#             dist_type=dx.MultivariateNormalDiag)
#         # Still want the output to be an array of dists at the moment
#         # self.distributions = pdfs_total
#         return ensemble_dist

#     def plot(self, ax=None, x: jnp.DeviceArray = None):
#         if not ax:
#             fig, ax = plt.subplots(figsize=(12, 5))
#         if x is None:
#             n_items = len(self.distributions)
#             x = jnp.arange(n_items)
#         means = jnp.asarray([e.mean() for e in self.distributions])
#         stddevs = jnp.asarray([e.stddev() for e in self.distributions])
#         ax.plot(x, means, label="Ensemble mean", color="tab:blue")
#         ax.fill_between(
#             x,
#             means - stddevs,
#             means + stddevs,
#             label="Ensemble one sigma",
#             color="tab:blue",
#             alpha=0.2,
#         )
#         return ax

class Barycentre(AbstractEnsembleScheme):
    def __init__(self, name: str = "Barycentre") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights: xr.DataArray, n_threads=2) -> Distribution:
        # if process_models[0].model_data.ndim > 2:
        #      raise NotImplementedError('Not implemented for more than temporal dimensions')
    
        bary_mus = []
        bary_stds = []
        n_points = int(process_models[0].model_data.size / process_models[0].model_data.realisation.size)
        weights = weights.values.reshape(process_models.number_of_models, -1)

        def _bary_calc(t_idx):
            means_t = []
            stds_t = []
            for process_model in process_models:
                if not process_model.distribution:
                    raise AttributeError(
                        f"No posterior for model {t_idx}. Please run model.fit() first."
                    )
                # Load pre-flattened distribution
                dist = process_model.distribution._dist
                mean = dist.mean()[t_idx]
                std = jnp.sqrt(dist.variance()[t_idx])
                means_t.append(mean)
                stds_t.append(std)
            weight = weights[:, t_idx]
            bary_mu, bary_std = gaussian_barycentre(np.asarray(means_t), np.asarray(stds_t), weight)

            return bary_mu, bary_std

        bary_output = Parallel(n_jobs=n_threads)(delayed(_bary_calc)(i) for i in trange(n_points))
        bary_output = np.asarray(bary_output)
        bary_mus = bary_output[:, 0]
        bary_stds = bary_output[:, 1]


        blank_array = xr.ones_like(process_models[0].model_data[0].drop('realisation')) * np.nan
        blank_array = blank_array.rename('blank')
        ensemble_dist = Distribution(
            mu=np.asarray(bary_mus), covariance=np.asarray(bary_stds) ** 2,
            dim_array=blank_array,
            dist_type=dx.MultivariateNormalDiag)
        return ensemble_dist

    def plot(self, ax=None, x: jnp.DeviceArray = None):
        if not ax:
            fig, ax = plt.subplots(figsize=(12, 5))
        if x is None:
            n_items = len(self.distributions)
            x = jnp.arange(n_items)
        means = jnp.asarray([e.mean() for e in self.distributions])
        stddevs = jnp.asarray([e.stddev() for e in self.distributions])
        ax.plot(x, means, label="Ensemble mean", color="tab:blue")
        ax.fill_between(
            x,
            means - stddevs,
            means + stddevs,
            label="Ensemble one sigma",
            color="tab:blue",
            alpha=0.2,
        )
        return ax


class MultiModelMean(AbstractEnsembleScheme):
    def __init__(self, name: str = "MultiModelMean") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights=None) -> jnp.DeviceArray:

        all_model_values = np.concatenate([pm.model_data.values for pm in process_models], axis=0)
        mean = np.mean(all_model_values, axis=0)
        std = np.std(all_model_values, axis=0)
        
        blank_array = xr.ones_like(process_models[0].model_data[0].drop('realisation')) * np.nan
        blank_array = blank_array.rename('blank')
        ensemble_dist = Distribution(
            mu=mean.ravel(), covariance=std.ravel() ** 2,
            dim_array=blank_array,
            dist_type=dx.MultivariateNormalDiag)
        return ensemble_dist


class WeightedModelMean(AbstractEnsembleScheme):
    def __init__(self, name: str = "MultiModelMean") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights: xr.DataArray) -> jnp.DeviceArray:
        weighted_mean = 0.0
        weighted_var = 0.0

        # TODO: Check weighted uncertainty calculations...
        for model in process_models:
            weight = weights.sel(model=model.model_name)
            model_mean = model.mean_across_realisations
            model_var = model.std_across_realisations ** 2
            weighted_var += model_var * (weight ** 2)
            weighted_mean += model_mean * weight

        blank_array = xr.ones_like(process_models[0].model_data[0].drop('realisation')) * np.nan
        blank_array = blank_array.rename('blank')
        ensemble_dist = Distribution(
            mu=weighted_mean.values.ravel(), covariance=weighted_var.values.ravel(),
            dim_array=blank_array,
            dist_type=dx.MultivariateNormalDiag)
        return ensemble_dist
