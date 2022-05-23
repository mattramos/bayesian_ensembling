from concurrent.futures import process
from copy import deepcopy
from selectors import EpollSelector
from importlib_metadata import distribution
from jax import jit
import typing as tp
import jax.numpy as jnp
from .data import ModelCollection, ProcessModel
import abc
from tqdm import trange, tqdm
import numpy as np
import copy
import xarray as xr

class AbstractWeight:
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:
        raise NotImplementedError

    def __call__(
        self, process_models: ModelCollection, observations: ProcessModel = None, **kwargs
    ) -> tp.Any:
        return self._compute(process_models=process_models, observations=observations, **kwargs)


class LogLikelihoodWeight(AbstractWeight):
    def __init__(self, name: str = "LogLikelihood") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel, return_lls=False
    ) -> jnp.DeviceArray:
        if process_models[0].model_data.ndim > 2:
             raise NotImplementedError('Not implemented for more than temporal dimensions')

        assert np.all(process_models.time == observations.time), "Time coordinates do not match between models and observations"
        model_lls = []
        for model in process_models:
            distribution = model.distribution
            # Expand dims is needed to ensure that the log_prob returns one point per time point
            log_likelihood = lambda x: distribution.log_prob(x)

            lls = []
            for obs_real in tqdm(observations):
                lls.append(log_likelihood(jnp.expand_dims(obs_real.values, -1)))
            lls_array = jnp.asarray(lls)
            lls_mean = jnp.mean(lls_array, axis=0)
            lls_mean_xarray = copy.deepcopy(model.model_data.isel(realisation=0)).drop_vars('realisation')
            lls_mean_xarray.data = lls_mean
            lls_mean_xarray = lls_mean_xarray.assign_coords(model=model.model_name)
            model_lls.append(lls_mean_xarray)

        # Put weights into an xarray DataArray for continuity and dimension description
        model_lls = xr.concat(model_lls, dim='model')  # (n_reals, time)
        weights = model_lls / model_lls.sum('model')

        assert weights.shape == (len(process_models), len(observations.time))

        if return_lls:
            return weights, model_lls
        else:
            return weights

class InverseSquareWeight(AbstractWeight):
    def __init__(self, name: str = "InverseSquareWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:

        weights = []
        for model in process_models:
            model_mean = model.mean_across_realisations
            obs_mean = observations.mean_across_realisations
            model_weight = (model_mean - obs_mean) ** -2
            model_weight = model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)
        
        weights = xr.concat(weights, dim='model')
        weights = weights / weights.sum('model')

        assert weights.time.size == model.time.size, "Weight is not the same size as model. Check observations and model time coordinates match!"

        return weights

class UniformWeight(AbstractWeight):
    def __init__(self, name: str = "InverseSquareWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:

        weights = []
        for model in process_models:
            model_weight = model.mean_across_realisations * 0 + 1. / len(process_models)
            model_weight = model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)
        
        weights = xr.concat(weights, dim='model')

        assert weights.time.size == model.time.size

        return weights