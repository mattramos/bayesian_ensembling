import distrax
import jax.numpy as jnp
import typing as tp
from dataclasses import dataclass
import numpy as np
from .array_types import ColumnVector
from .plotters import _unique_legend, get_style_cycler
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import warnings
sns.set_style('darkgrid')


@dataclass
class ProcessModel:
    model_data: xr.DataArray
    model_name: str
    idx: int = 0
    _distribution = None

    def __post_init__(self):
        self.model_mean = self.model_data.mean()
        self.model_std = self.model_data.std()
        self.climatology = None
        # TODO: Do some check that time and real are in the data
        # We want a specific order of coords (real, time, space) and we want specific names

        # Could guess the var name if a dataset is given

    # @property
    # def realisations(self) -> jnp.DeviceArray:
    #     assert "time" not in self.model_data.columns
    #     return self.model_data.values

    @property
    def max_val(self) -> int:
        return self.model_data.max()

    @property
    def min_val(self) -> int:
        return self.model_data.min()

    @property
    def n_observations(self) -> int:
        raise NotImplementedError
        return self.model_data.shape[0]

    @property
    def n_realisations(self) -> int:
        return self.model_data.realisation.size

    @property
    def time(self) -> ColumnVector:
        time = self.model_data.time
        return time

    def standardise_data(self):
        name = self.model_name + ' standardised'
        standardised_data = (self.model_data - self.model_mean) / self.model_std
        standardised_model = ProcessModel(standardised_data , name)
        assert not hasattr(self, 'original_mean'), "This data is already standardised!"
        standardised_model.original_mean = self.model_mean
        standardised_model.original_std = self.model_std
        return standardised_model

    def unstandardise_data(self):
        assert hasattr(self, 'original_mean'), "This data is not standardised!"
        name = self.model_name + ' unstandardised'
        unstandardised_data = self.model_data * self.original_std + self.original_mean
        unstandardised_model = ProcessModel(unstandardised_data , name)
        return unstandardised_model

    def calculate_anomaly(self, climatology_dates=["1961-01-01", "1990-12-31"], climatology=False):
        # If a climatology is not specified, it calculates one and returns it
        da = self.model_data.copy(deep=True)
        if np.any(climatology) == False:
            t0 = climatology_dates[0]
            t1 = climatology_dates[1]
            clim_years = da.sel(time=slice(t0, t1))
            clim = clim_years.groupby('time.month').mean().mean('realisation')
        else:
            clim = climatology
            assert clim.month.size == 12, 'Climatology is the incorrect length (must be 12)'
        da_anom = da.groupby('time.month') - clim
        # Save climatology
        anomaly_model = ProcessModel(da_anom, self.model_name + ' anomaly')
        anomaly_model.climatology = clim

        return anomaly_model

    def plot(self, **kwargs) -> tp.Any:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = self.model_data.time
        if self.model_data.ndim > 2:
            warnings.warn('Collapsing (mean) non-time dimensions for plotting')
            coord_names = [coord for coord in self.model_data.coords]
            coord_names.remove('time')
            coord_names.remove('realisation')
            da = self.model_data.mean(coord_names)
        else:
            da = self.model_data
        ax.set_prop_cycle(get_style_cycler())
        for real in da.realisation:
            ax.plot(x, da.sel(realisation=real), alpha=0.1, color='gray', label='Realisations', ls='-')
        ax.plot(x, da.mean('realisation'), label='Model mean', alpha=0.7)
        ax.legend(loc='best')
        ax = _unique_legend(ax)
        ax.set_title(self.model_name)
        return ax

    @property
    def mean_across_realisations(self):
        return self.model_data.mean('realisation')

    @property
    def std_across_realisations(self):
        return self.model_data.std('realisation')

    @property
<<<<<<< HEAD
    def distribution(self) -> distrax.Distribution:
        return self._distribution
=======
    def ndim(self):
        return self.model_data.ndim

    # @property
    # # TODO: Implement this in more than a time dimension...?
    # # TODO: Merge with Tom's as distribution
    # def as_multivariate_gaussian(self) -> distrax.Distribution:
    #     if self.ndim > 2:
    #         raise NotImplementedError, "No implementation for 3D data yet"
    #     else:
    #         L = np.linalg.cholesky(self.temporal_covariance + jnp.eye(self.n_observations) * 1e-8)
    #         return distrax.MultivariateNormalTri(self.temporal_mean, L)

    @property
    def distribution(self) -> distrax.Distribution:
        if self.ndim > 2:
            raise NotImplementedError, "No implementation for 3D data yet"
        else:
            return self._distribution
>>>>>>> xarray update

    @distribution.setter
    def distribution(self, mvn: distrax.Distribution):
        self._distribution = mvn

    def __len__(self) -> int:
        return self.n_realisations

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out = self.model_data.isel(realisation=self.idx)
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
        try:
            out = self.models[self.idx]
            self.idx += 1
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.
        return out

    @property
    def time(self) -> ColumnVector:
        return self.models[0].time

    @property
    def max_val(self) -> int:
        return np.max([model.max_val for model in self.models])

    @property
    def min_val(self) -> int:
        return np.min([model.min_val for model in self.models])

    # @property
    # def n_observations(self) -> int:
    #     return self.models[0].model_data.shape[0]

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

<<<<<<< HEAD
    def plot_all(self, ax: tp.Optional[tp.Any] = None, legend: bool = False, **kwargs) -> tp.Any:
        if not ax:
            fig, ax = plt.subplots(figsize=(15, 7))
=======
    # TODO: tie in with whatever is decided above RE distributions
    # def multivariate_gaussian_set(self) -> tp.Dict[str, distrax.Distribution]:
    #     return {model.model_name: model.as_multivariate_gaussian for model in self.models}
>>>>>>> xarray update

        ax.set_prop_cycle(get_style_cycler())
        for model in self:
            if model.model_data.ndim > 2:
                warnings.warn('Collapsing (mean) non-time dimensions for plotting')
                coord_names = [coord for coord in model.model_data.coords]
                coord_names.remove('time')
                da = model.model_data.mean(coord_names)
            else:
                da = model.model_data.mean('realisation')
            x = model.time
            ax.plot(x, da, alpha=0.5, label=model.model_name)
        ax.legend(loc='best')
        fig.show()

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
            if model.model_data.ndim > 2:
                warnings.warn('Collapsing (mean) non-time dimensions for plotting')
                coord_names = [coord for coord in model.model_data.coords]
                coord_names.remove('time')
                model_mean = model.model_data.mean(coord_names)
                coord_names.remove('realisation')
                reals = [real.mean(coord_names) for real in model]
            else:
                reals = [real for real in model]
                model_mean = model.mean_across_realisations
            
            ax.plot(x, model_mean, alpha=0.7, label=model.model_name, **args, zorder=10)
            
            [ax.plot(x, real, alpha=0.1, color='gray', label='Realisations', zorder=1) for real in reals]
            ax.legend(loc='best')
            ax = _unique_legend(ax)

        fig.show()

    @property
    def covariance_stack(self) -> jnp.DeviceArray:
        return jnp.stack([model.distribution.covariance() for model in self.models])

    @property
    def mean_stack(self) -> jnp.DeviceArray:
        return jnp.stack([model.distribution.mean() for model in self.models])
