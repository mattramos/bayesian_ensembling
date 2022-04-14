from importlib_metadata import distribution
from jax import jit
import typing as tp
import jax.numpy as jnp
from .data import ModelCollection, ProcessModel
import abc


class Weight:
    def __init__(self, name: str) -> None:
        self.name

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> jnp.DeviceArray:
        raise NotImplementedError

    def __call__(
        self,
        process_models: ModelCollection,
        observations: ProcessModel,
    ) -> tp.Any:
        return self._compute(process_model=process_models, observations=observations)


class LogLikelhoodWeight(Weight):
    def __init__(self, name: str = "LogLikelihood") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> jnp.DeviceArray:
        for model in process_models:
            distribution = model.as_multivariate_gaussian
            log_likelihood = jit(lambda x: distribution.log_prob(x))
            for obs in observations:
                lls = log_likelihood(obs.reshape(-1, 1))
