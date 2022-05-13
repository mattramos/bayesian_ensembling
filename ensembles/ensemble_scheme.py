from jax import jit
import typing as tp
import jax.numpy as jnp
from .data import ModelCollection, ProcessModel
from .wasserstein import gaussian_barycentre
import abc
from tqdm import trange, tqdm
import numpy as np
import ot
import distrax
import matplotlib.pyplot as plt


class AbstractEnsembleScheme:
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    # TODO define a union array type of jnp and np arrays
    def _compute(self, process_models: ModelCollection) -> jnp.DeviceArray:
        raise NotImplementedError

    def __call__(self, process_models: ModelCollection, **kwargs) -> tp.Any:
        return self._compute(process_models=process_models, **kwargs)


class Barycentre(AbstractEnsembleScheme):
    def __init__(self, name: str = "Barycentre") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights: np.ndarray) -> jnp.DeviceArray:

        mvns = process_models.distributions()
        # TODO: use jax for these loops
        # TODO: calculate sensible support limits

        pdfs_total = []
        for t_idx in trange(process_models[0].n_observations):
            means_t = []
            stds_t = []
            for name, mvn in mvns.items():
                mean = mvn.mean()[t_idx]
                std = jnp.sqrt(mvn.variance()[t_idx])
                means_t.append(mean)
                stds_t.append(std)
            weight = np.array(weights[t_idx])
            bary_mu, bary_std = gaussian_barycentre(np.asarray(means_t), np.asarray(stds_t), weight)

            pdfs_total.append(distrax.Normal(bary_mu, bary_std))
        # Still want the output to be an array of dists at the moment
        return np.asarray(pdfs_total)


class MultiModelMean(AbstractEnsembleScheme):
    def __init__(self, name: str = "MultiModelMean") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection) -> jnp.DeviceArray:
        all_model_values = np.concatenate([pm.model_data.values for pm in process_models], axis=1)
        mean = np.mean(all_model_values, axis=1)
        std = np.std(all_model_values, axis=1)
        return np.asarray([distrax.Normal(m, s) for m, s in zip(mean, std)])


class WeightedModelMean(AbstractEnsembleScheme):
    def __init__(self, name: str = "MultiModelMean") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights: jnp.ndarray) -> jnp.DeviceArray:
        assert np.logical_and(
            weights.ndim == 1, weights.shape[0] == process_models.number_of_models
        ), "Weights must be 1D and have the same length as the number of models"
        weighted_mean = 0.0
        weighted_var = 0.0

        for weight, model in zip(weights, process_models):
            model_mean = model.temporal_mean
            model_var = model.temporal_covariance.diagonal()
            weighted_var += model_var * (weight ** 2)
            weighted_mean += model_mean * weight

        return np.asarray(
            [distrax.Normal(m, s) for m, s in zip(weighted_mean, jnp.power(weighted_var, 0.5))]
        )
