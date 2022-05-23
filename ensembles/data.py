from importlib.util import module_for_loader
from turtle import pos
import distrax
import pandas as pd
import jax.numpy as jnp
import tensorflow as tf
import typing as tp
from dataclasses import dataclass
import numpy as np

from ensembles.models import AbstractModel
from .array_types import ColumnVector, Matrix
from .plotters import _unique_legend, cmap, get_style_cycler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


@dataclass
class Dataset:
    Xs: tp.List[Matrix]
    y: ColumnVector

    def __post_init__(self):
        assert isinstance(self.Xs, list), "Input data must be a list"
        for dataset in self.Xs:
            tf.debugging.assert_shapes([(dataset, ("N", "D")), (self.y, ("N", 1))])

    @property
    def n_datasets(self) -> int:
        return len(self.Xs)

    @property
    def n(self) -> int:
        return self.y.shape[0]

    def __len__(self) -> int:
        return self.y.shape[0]


@dataclass
class ProcessModel:
    model_data: pd.DataFrame
    model_name: str
    idx: int = 0
    _distribution = None

    def __post_init__(self):
        self.model_mean = self.model_data.mean()
        self.model_std = self.model_data.std()
        self.climatology = None

    @property
    def realisations(self) -> jnp.DeviceArray:
        assert "time" not in self.model_data.columns
        return self.model_data.values

    @property
    def max_val(self) -> int:
        return np.max(self.model_data.values)

    @property
    def min_val(self) -> int:
        return np.min(self.model_data.values)

    @property
    def n_observations(self) -> int:
        return self.model_data.shape[0]

    @property
    def n_realisations(self) -> int:
        return self.model_data.shape[1]

    @property
    def time(self) -> ColumnVector:
        time = pd.DatetimeIndex(self.model_data.index)
        return time

    def standardise_data(self) -> pd.DataFrame:
        # TODO: Return a ProcessModel from here
        # TODO: May have to think about standardising the data before/after climatology at the ensemble level.
        return self.model_data.sub(self.model_mean).div(self.model_std)

    def unstandardise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Return a ProcessModel from here
        return df.mul(self.model_std).add(self.model_mean)

    def calculate_anomaly(self, climatology=False, resample_freq=None):
        # If a climatology is not specified, it calculates one and returns it
        df_ = self.model_data.copy(deep=True)
        if np.any(climatology) == False:
            t0 = "1961-01-01"
            t1 = "1990-31-12"
            df_["month"] = [int(t.split("-")[1]) for t in df_.index]
            clim_df = df_[np.logical_and(df_.index >= t0, df_.index <= t1)]
            clim = clim_df.groupby(["month"]).mean().mean(axis=1).values
            df_.drop(["month"], axis=1, inplace=True)
        else:
            clim = climatology
            assert clim.shape == (12,), "Climatology is the incorrect length (must be 12)"
        clim_tot = np.tile(clim, len(df_) // 12).reshape(-1, 1)
        df_ = df_ - clim_tot
        # Save climatology
        self.climatology = clim
        if resample_freq:
            df_ = df_.set_index(pd.DatetimeIndex(df_.index)).resample(resample_freq).mean()
        anomaly_model = ProcessModel(df_, self.model_name)
        anomaly_model.climatology = clim

        return anomaly_model

    # def calculate_climatology(self) -> pd.DataFrame:
    #     # TODO: Return a ProcessModel from here
    #     # TODO: Matt to tidy this up
    #     df = self.model_data.copy(deep=True)
    #     t0 = "1961-01-01"
    #     t1 = "1990-31-12"
    #     df["month"] = [int(t.split("-")[1]) for t in df.index]
    #     for col in df.columns[:-1]:
    #         clim_df = df[[col, "month"]][np.logical_and(df.index >= t0, df.index <= t1)]
    #         clim = clim_df.groupby(["month"]).mean()[col].values
    #         clim_tot = np.tile(clim, len(df) // 12)
    #         df[col] = df[col] - clim_tot
    #     df.drop(["month"], axis=1, inplace=True)
    #     return df

    def plot(self, ax: tp.Optional[tp.Any] = None, **kwargs) -> tp.Any:
        # TODO: Write some plotting code here.
        if not ax:
            fig, ax = plt.subplots(figsize=(12, 7))
        x = self.time
        ax.set_prop_cycle(get_style_cycler())
        ax.plot(x, self.realisations, alpha=0.1, color="gray", label="Realisations")
        ax.plot(x, self.temporal_mean, label="Model mean", alpha=0.7)
        ax.legend(loc="best")
        ax = _unique_legend(ax)
        ax.set_title(self.model_name)
        return ax

    @property
    def temporal_mean(self) -> jnp.DeviceArray:
        model_vals = self.realisations
        return jnp.mean(model_vals, axis=1)

    @property
    def temporal_covariance(self) -> jnp.DeviceArray:
        # TODO: Write shape checking unit test for this.
        model_vals = self.realisations
        return jnp.cov(model_vals)

    @property
    def distribution(self) -> distrax.Distribution:
        return self._distribution

    @distribution.setter
    def distribution(self, mvn: distrax.Distribution):
        self._distribution = mvn

    def __len__(self) -> int:
        return self.n_realisations

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out = self.realisations[:, self.idx]
            self.idx += 1
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.
        return out


@dataclass
class ModelCollection:
    models: tp.List[ProcessModel]
    idx: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: Check this after MA change
        try:
            out = self.models[self.idx]
            self.idx += 1
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.
        return out

    def fit(self, model: AbstractModel, **kwargs):
        for process_model in self.models:
            posterior = model.fit(process_model, **kwargs)
            process_model.distribution = posterior

    @property
    def time(self) -> ColumnVector:
        return self.models[0].time

    @property
    def max_val(self) -> int:
        return np.max([model.max for model in self.models])

    @property
    def min_val(self) -> int:
        return np.min([model.max for model in self.models])

    @property
    def n_observations(self) -> int:
        return self.models[0].model_data.shape[0]

    @property
    def number_of_models(self):
        return len(self.models)

    @property
    def model_names(self):
        return [model.model_name for model in self.models]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, item):
        return self.models[item]

    def distributions(self) -> tp.Dict[str, distrax.Distribution]:
        return {model.model_name: model.distribution for model in self.models}

    def plot_all(
        self, ax: tp.Optional[tp.Any] = None, legend: bool = False, one_color: str = None, **kwargs
    ) -> tp.Any:
        if not ax:
            fig, ax = plt.subplots(figsize=(15, 7))

        ax.set_prop_cycle(get_style_cycler())
        for model in self:
            x = model.time
            if one_color:
                ax.plot(x, model.temporal_mean, alpha=0.3, color=one_color)
            else:
                ax.plot(x, model.temporal_mean, alpha=0.8, label=model.model_name)
        if legend:
            ax.legend(loc="best")
        return ax

    def plot_posteriors(
        self, ax: tp.Optional[tp.Any] = None, legend: bool = False, one_color: str = None, **kwargs
    ) -> tp.Any:
        if not ax:
            fig, ax = plt.subplots(figsize=(15, 7))

        ax.set_prop_cycle(get_style_cycler())
        x = self.models[0].time
        for model_name, dist in self.distributions().items():
            if one_color:
                ax.plot(x, dist.mean(), alpha=0.3, color=one_color)
                ax.fill_between(
                    x,
                    dist.mean() - dist.stddev(),
                    dist.mean() + dist.stddev(),
                    alpha=0.3,
                    color=one_color,
                )
            else:
                ax.plot(x, jnp.asarray(dist.mean()), alpha=0.3, label=model_name)
                ax.fill_between(
                    x,
                    jnp.asarray(dist.mean()) - jnp.asarray(dist.stddev()),
                    jnp.asarray(dist.mean()) + jnp.asarray(dist.stddev()),
                    alpha=0.3,
                )
        if legend:
            ax.legend(loc="best")
        return ax

    def plot_grid(self, ax: tp.Optional[tp.Any] = None, **kwargs) -> tp.Any:
        style_cycler = get_style_cycler()
        fig, axes = plt.subplots(
            figsize=(15, 4 * np.ceil(self.number_of_models / 3)),
            nrows=round(np.ceil(self.number_of_models / 3)),
            ncols=3,
            sharey=True,
        )
        for model, ax, args in zip(self, axes.ravel(), style_cycler):
            x = model.time
            ax.plot(x, model.realisations, alpha=0.1, color="gray", label="Realisations")
            ax.plot(x, model.temporal_mean, alpha=0.7, label=model.model_name, **args)
            ax.legend(loc="best")
            ax = _unique_legend(ax)

        fig.show()

    @property
    def covariance_stack(self) -> jnp.DeviceArray:
        return jnp.stack([model.distribution.covariance() for model in self.models])

    @property
    def mean_stack(self) -> jnp.DeviceArray:
        return jnp.stack([model.distribution.mean() for model in self.models])
