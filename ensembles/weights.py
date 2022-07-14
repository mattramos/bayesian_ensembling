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
import distrax as dx


class AbstractWeight:
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:
        raise NotImplementedError

    def __call__(
        self,
        process_models: ModelCollection,
        observations: ProcessModel = None,
        **kwargs
    ) -> tp.Any:
        return self._compute(
            process_models=process_models, observations=observations, **kwargs
        )


class LogLikelihoodWeight(AbstractWeight):
    def __init__(self, name: str = "LogLikelihood") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self,
        process_models: ModelCollection,
        observations: ProcessModel,
        return_lls=False,
        standardisation_scheme="exp",
        standardisation_constant=0.01,
    ) -> jnp.DeviceArray:
        # if process_models[0].model_data.ndim > 2:
        #      raise NotImplementedError('Not implemented for more than temporal dimensions')

        assert np.all(
            process_models.time == observations.time
        ), "Time coordinates do not match between models and observations"
        model_lls = []
        for model in process_models:
            distribution = model.distribution._dist
            # Expand dims is needed to ensure that the log_prob returns one point per time point
            log_likelihood = lambda x: distribution.log_prob(x)

            lls = []
            for obs_real in tqdm(observations):
                # assert distribution.event_shape == obs_real.values.ravel().shape, 'Observations are not the same size as the model distribution'
                # This is need because dx.Normal and dx.MultiVariate treat inputs differently
                if model.distribution.dist_type == dx.Normal:
                    ll_val = log_likelihood(obs_real.values.ravel())
                else:
                    ll_val = log_likelihood(
                        jnp.expand_dims(obs_real.values.ravel(), -1)
                    )
                lls.append(ll_val)

            lls_array = jnp.asarray(lls)
            lls_mean = jnp.mean(lls_array, axis=0)
            # TODO: Question about whether these should be done on the mean or on the individual log-likelihoods?
            if standardisation_scheme == "exp":
                # Exponentially scales - enforces positivity
                # But this adds an arbitrary constant...?
                lls_mean = np.exp(standardisation_constant * lls_mean)
            # elif standardisation_scheme == 'min-max':
            #     # Scales everything from 0 to 1
            #     lls_mean = (lls_mean - np.nanmin(lls_mean)) / (np.nanmax(lls_mean) - np.nanmin(lls_mean))
            # elif standardisation_scheme == 'subtract-constant':
            #     # Here this ensures everything is negative
            #     lls_mean = lls_mean - np.ceil(lls_mean)
            lls_mean_xarray = copy.deepcopy(
                model.model_data.isel(realisation=0)
            ).drop_vars("realisation")
            lls_mean_xarray.data = lls_mean.reshape(lls_mean_xarray.shape)
            lls_mean_xarray = lls_mean_xarray.assign_coords(model=model.model_name)

            model_lls.append(lls_mean_xarray)

        # Put weights into an xarray DataArray for continuity and dimension description
        model_lls = xr.concat(model_lls, dim="model")  # (n_reals, time)
        model_lls = model_lls.rename("Log-likelihoods")
        model_lls_sum = model_lls.sum(
            "model"
        )  # This returns zero where there were nans
        # model_lls_sum_masked = model_lls_sum.where(model_lls_sum != 0.000) # Need this to replace 0.000s with nans
        weights = model_lls / model_lls_sum
        # weights = model_lls / model_lls_sum_masked

        weights = weights.rename("Log-likelihood weights")

        assert weights.shape == (len(process_models),) + obs_real.shape

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

        weights = xr.concat(weights, dim="model")
        weights = weights / weights.sum("model")

        assert (
            weights.time.size == model.time.size
        ), "Weight is not the same size as model. Check observations and model time coordinates match!"

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
            model_weight = model.mean_across_realisations * 0 + 1.0 / len(
                process_models
            )
            model_weight = model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)

        weights = xr.concat(weights, dim="model")

        assert weights.time.size == model.time.size

        return weights
