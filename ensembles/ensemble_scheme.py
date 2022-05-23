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
import xarray as xr


class AbstractEnsembleScheme:
    def __init__(self, name: str) -> None:
        self.name = name
        self.distributions = None

    @abc.abstractmethod
    # TODO define a union array type of jnp and np arrays
    def _compute(
        self, process_models: ModelCollection, weights: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        raise NotImplementedError

    def __call__(
        self, process_models: ModelCollection, weights: jnp.DeviceArray, **kwargs
    ) -> tp.Any:
        return self._compute(process_models=process_models, weights=weights, **kwargs)


class Barycentre(AbstractEnsembleScheme):
    def __init__(self, name: str = "Barycentre") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights: xr.DataArray) -> jnp.DeviceArray:
        if process_models[0].model_data.ndim > 2:
             raise NotImplementedError('Not implemented for more than temporal dimensions')
        
        mvns = process_models.distributions()
        # TODO: use jax for these loops
        # TODO: calculate sensible support limits

        pdfs_total = []
        for t_idx in trange(len(process_models[0].time)):
            means_t = []
            stds_t = []
            for name, mvn in mvns.items():
                if not mvn:
                    raise AttributeError(
                        f"No posterior for model {t_idx}. Please run model.fit() first."
                    )
                mean = mvn.mean()[t_idx]
                std = jnp.sqrt(mvn.variance()[t_idx])
                means_t.append(mean)
                stds_t.append(std)
            weight = np.asarray(weights.isel(time=t_idx).values)
            bary_mu, bary_std = gaussian_barycentre(np.asarray(means_t), np.asarray(stds_t), weight)

            pdfs_total.append(distrax.Normal(bary_mu, bary_std))
        # Still want the output to be an array of dists at the moment
        self.distributions = pdfs_total
        return np.asarray(pdfs_total)

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
    def _compute(self, process_models: ModelCollection) -> jnp.DeviceArray:
        # TODO: Could write this for xarray?
        all_model_values = np.concatenate([pm.model_data.values for pm in process_models], axis=0)
        mean = np.mean(all_model_values, axis=0)
        std = np.std(all_model_values, axis=0)
        return np.asarray([distrax.Normal(m, s) for m, s in zip(mean, std)])


class WeightedModelMean(AbstractEnsembleScheme):
    def __init__(self, name: str = "MultiModelMean") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(self, process_models: ModelCollection, weights: xr.DataArray) -> jnp.DeviceArray:
        weighted_mean = 0.0
        weighted_var = 0.0

        for model in process_models:
            weight = weights.sel(model=model.model_name)
            model_mean = model.mean_across_realisations
            model_var = model.std_across_realisations ** 2
            weighted_var += model_var * (weight ** 2)
            weighted_mean += model_mean * weight

        return np.asarray(
            [distrax.Normal(m, s) for m, s in zip(weighted_mean.values, jnp.power(weighted_var.values, 0.5))]
        )
