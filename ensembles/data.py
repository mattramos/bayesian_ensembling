from importlib.util import module_for_loader
import distrax
import pandas as pd
import jax.numpy as jnp
import tensorflow as tf
import typing as tp
from dataclasses import dataclass
import numpy as np
from .array_types import ColumnVector, Matrix
from .plotters import _unique_legend, cmap, get_style_cycler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set_style('darkgrid')


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

    def __post_init__(self):
        self.model_mean = self.model_data.mean()
        self.model_std = self.model_data.std()

    @property
    def realisations(self) -> jnp.DeviceArray:
        assert "time" not in self.model_data.columns
        return self.model_data.values

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

    def calculate_climatology(self) -> pd.DataFrame:
        # TODO: Return a ProcessModel from here
        # TODO: Matt to tidy this up
        df = self.model_data.copy(deep=True)
        t0 = "1961-01-01"
        t1 = "1990-31-12"
        df["month"] = [int(t.split("-")[1]) for t in df.index]
        for col in df.columns[:-1]:
            clim_df = df[[col, "month"]][np.logical_and(df.index >= t0, df.index <= t1)]
            clim = clim_df.groupby(["month"]).mean()[col].values
            clim_tot = np.tile(clim, len(df) // 12)
            df[col] = df[col] - clim_tot
        df.drop(["month"], axis=1, inplace=True)
        return df

    def plot(self, ax: tp.Optional[tp.Any] = None, **kwargs) -> tp.Any:
        # TODO: Write some plotting code here.
        fig, ax = plt.subplots(figsize=(12, 7))
        x = self.time
        ax.set_prop_cycle(style_cyler())
        ax.plot(x, self.realisations, alpha=0.1, color='gray', label='Realisations')
        ax.plot(x, self.temporal_mean, label='Model mean', alpha=0.7)
        ax.legend(loc='best')
        ax = _unique_legend(ax)
        ax.set_title(self.model_name)
        fig.show()

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
    def as_multivariate_gaussian(self) -> distrax.Distribution:
        L = jnp.linalg.cholesky(self.temporal_covariance + jnp.eye(self.n_observations) * 1e-8)
        return distrax.MultivariateNormalTri(self.temporal_mean, L)

    def __len__(self) -> int:
        return self.n_realisations

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out =  self.realisations[:, self.idx]
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
            out =  self.models[self.idx]
            self.idx += 1
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.
        return out

    @property
    def number_of_models(self):
        return len(self.models)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, item):
        return self.models[item]

    def multivariate_gaussian_set(self) -> tp.Dict[str, distrax.Distribution]:
        return {model.model_name: model.as_multivariate_gaussian for model in self.models}

    def plot_all(self, ax: tp.Optional[tp.Any] = None, **kwargs) -> tp.Any:
        # TODO: Write some plotting code here.
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_prop_cycle(get_style_cycler())
        for model in self:
            x = model.time
            ax.plot(x, model.temporal_mean, alpha=0.5, label=model.model_name)
        ax.legend(loc='best')
        fig.show()

    def plot_grid(self, ax: tp.Optional[tp.Any] = None, **kwargs) -> tp.Any:
        # TODO: Write some plotting code here.
        style_cycler = get_style_cycler()
        fig, axes = plt.subplots(
            figsize=(10, 4 * np.ceil(self.number_of_models/3)),
            nrows=round(np.ceil(self.number_of_models/3)),
            ncols=3,
            sharey=True)
        for model, ax, args in zip(self, axes.ravel(), style_cycler):
            x = model.time
            ax.plot(x, model.realisations, alpha=0.1, color='gray', label='Realisations')
            ax.plot(x, model.temporal_mean, alpha=0.7, label=model.model_name, **args)
            ax.legend(loc='best')
            ax = _unique_legend(ax)

        fig.show()