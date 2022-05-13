from concurrent.futures import process
from selectors import EpollSelector
from importlib_metadata import distribution
from jax import jit
import typing as tp
import jax.numpy as jnp
from .data import ModelCollection, ProcessModel
import abc
from tqdm import trange, tqdm
import numpy as np
import xarray as xr

class Weight:
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> jnp.DeviceArray:
        raise NotImplementedError

    def __call__(
        self,
        process_models: ModelCollection,
        observations: ProcessModel,
        **kwargs
    ) -> tp.Any:
        return self._compute(process_models=process_models, observations=observations, **kwargs)


class LogLikelihoodWeight(Weight):
    def __init__(self, name: str = "LogLikelihood") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection,
        observations: ProcessModel,
        return_lls=False
    ) -> jnp.DeviceArray:
        model_lls = []
        for model in process_models:
            distribution = model.as_multivariate_gaussian
            # log_likelihood = jit(lambda x: distribution.log_prob(x))
            log_likelihood = lambda x: distribution.log_prob(x)

            lls = []
            for obs_real in tqdm(observations):
                lls.append(log_likelihood(obs_real.reshape(-1, 1)))
            lls_array = jnp.asarray(lls)
            lls_mean = jnp.mean(lls_array, axis=0)
            model_lls.append(lls_mean)
        
        model_lls = jnp.asarray(model_lls).T # (time, n_reals)

        weights = (model_lls / jnp.expand_dims(jnp.sum(model_lls, axis=1), -1))

        assert weights.shape == (observations.n_observations, len(process_models))
        #TODO: implement this properly
        # assert np.all(np.testing.assert_almost_equal(weights, 1.))

        if return_lls:
            return weights, model_lls
        else:
            return weights

class InverseSquareWeight(Weight):
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
            model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)
        
        weights = xr.concat(weights, dim='model')
        weights = weights / weights.sum('model')

        return weights

class UniformWeight(Weight):
    def __init__(self, name: str = "InverseSquareWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:

        weights = []
        for model in process_models:
            model_weight = model.mean_across_realisations * 0 + 1. / len(process_models)
            model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)
        
        weights = xr.concat(weights, dim='model')

        return weights