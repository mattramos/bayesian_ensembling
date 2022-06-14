from copy import copy
from distutils import dist
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

# from .models import AbstractModel
import distrax as dx
import cartopy.crs as ccrs
import jax.random as jr

key = jr.PRNGKey(123)


@dataclass
class Distribution:
    mu: np.array
    covariance: np.array
    dim_array: xr.DataArray
    dist_type: dx.Distribution

    def __post_init__(self):
        self._dist = self.dist_type(self.mu, self.covariance)

    def reshape(self, vals, name=False):
        reshaped_vals = vals.reshape(self.dim_array.shape)
        reshaped_array = self.dim_array.copy(data=reshaped_vals)
        if name:
            reshaped_array = reshaped_array.rename(name)
        return reshaped_array

    # TODO: Could add in xarray indexing args through an arg dict for xarray sel
    def plot_temporally(self):
        reshaped_mean = self.reshape(self._dist.mean(), name="Distribution mean")
        reshaped_sigma = np.sqrt(
            self.reshape(self._dist.variance(), name="Variance mean")
        )
        if reshaped_mean.ndim > 2:
            warnings.warn("Collapsing (mean) non-time dimensions for plotting")
            coord_names = [coord for coord in reshaped_mean.coords]
            coord_names.remove("time")
            reshaped_mean = reshaped_mean.mean(coord_names)
            reshaped_sigma = reshaped_sigma.mean(coord_names)

        mean = reshaped_mean.values
        sig = reshaped_sigma.values
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.fill_between(
            reshaped_mean.time.values,
            mean - sig,
            mean + sig,
            alpha=0.2,
            color="tab:blue",
        )
        ax.fill_between(
            reshaped_mean.time.values,
            mean - 2 * sig,
            mean + 2 * sig,
            alpha=0.2,
            color="tab:blue",
        )
        ax.fill_between(
            reshaped_mean.time.values,
            mean - 3 * sig,
            mean + 3 * sig,
            alpha=0.2,
            color="tab:blue",
        )
        ax.plot(reshaped_mean.time.values, mean, color="tab:blue", zorder=10)
        fig.show()

    def plot_spatially(self):
        reshaped_mean = self.mean
        reshaped_sigma = np.sqrt(self.variance)
        # TODO: could add in area weighting into this
        if "time" in reshaped_mean.coords:
            reshaped_mean = reshaped_mean.mean("time")
            reshaped_sigma = reshaped_sigma.mean("time")
            warnings.warn("Collapsing (mean) temporal dimensions for plotting")
        mean = reshaped_mean
        sig = reshaped_sigma
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12, 7),
            subplot_kw=dict(projection=ccrs.Robinson()),
        )
        mean.plot(
            ax=axes[0],
            cbar_kwargs={"orientation": "horizontal"},
            transform=ccrs.PlateCarree(),
        )
        sig.plot(
            ax=axes[1],
            cbar_kwargs={"orientation": "horizontal"},
            transform=ccrs.PlateCarree(),
        )
        axes[0].coastlines()
        axes[1].coastlines()
        fig.tight_layout()
        fig.show()

    def plot(self, ax=None, **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(15, 7))

        ax.set_prop_cycle(get_style_cycler())
        mean = self.mean
        var = self.variance
        if mean.ndim > 1:
            warnings.warn("Collapsing (mean) non-time dimensions for plotting")
            coord_names = [coord for coord in mean.coords]
            coord_names.remove("time")
            mean = mean.mean(coord_names)
            var = var.mean(coord_names)
        x = mean.time.values
        ax.fill_between(
            x,
            mean.values - np.sqrt(var.values),
            mean.values + np.sqrt(var.values),
            alpha=0.2,
            color="tab:blue",
        )
        ax.plot(x, mean.values, color="tab:blue")
        return ax

    @property
    def mean(self):
        return self.reshape(self._dist.mean(), name="Distribution mean")

    @property
    def variance(self):
        return self.reshape(self._dist.variance(), name="Distribution variance")

    def sample(self):
        samples = np.asarray(self._dist.sample(seed=np.random.randint(0, 110000)))
        return self.reshape(samples, name="Distribution sample")


@dataclass
class ProcessModel:
    """Data class for handing the simulation output for a single process model.

    Args:
        model_data (xr.DataArray): The input model data. Realisation must be the first dimension and time must be the second.
        model_name (str): The model name. It should be unique to the model.

    Returns:
        ProcessModel: A data class for handling all the data for a singel process model.
    """

    model_data: xr.DataArray
    model_name: str
    idx: int = 0
    _distribution = None

    def __post_init__(self):
        self.model_mean = self.model_data.mean()
        self.model_std = self.model_data.std()
        self.climatology = None
        assert isinstance(self.model_data, xr.DataArray), "Input must be xr.DataArray"
        assert self.model_data.dims[0] == "realisation"
        # TODO: Do some check that time and real are in the data
        # We want a specific order of coords (real, time, space) and we want specific names

    @property
    def max_val(self) -> int:
        """Returns the maximum value of the process models output

        Returns:
            int: The maximum value
        """
        return self.model_data.max()

    @property
    def min_val(self) -> int:
        """Returns the minimum value of the process models output

        Returns:
            int: The minimum value
        """
        return self.model_data.min()

    @property
    def n_observations(self) -> int:
        """Returns the number of observation data points

        Raises:
            NotImplementedError: Currently not used as unsure of use in 3D case

        Returns:
            int: Number of observations
        """
        raise NotImplementedError
        return self.model_data.shape[0]

    @property
    def n_realisations(self) -> int:
        """Returns the number of realisations within the process model

        Returns:
            int: Number of realisations
        """
        return self.model_data.realisation.size

    @property
    def time(self) -> xr.DataArray:
        """Returns the time dimension which is useful for plotting and comparing to other models and observations. Note
        this doesn't work for plotting with plt.fill_between(...). For this purpose, use ProcessModel.time.values.

        Returns:
            xr.DataArray: A data array object of the time dimension
        """
        time = self.model_data.time
        return time

    def standardise_data(self):
        name = self.model_name + " standardised"
        standardised_data = (self.model_data - self.model_mean) / self.model_std
        standardised_model = ProcessModel(standardised_data, name)
        assert not hasattr(self, "original_mean"), "This data is already standardised!"
        standardised_model.original_mean = self.model_mean
        standardised_model.original_std = self.model_std
        return standardised_model

    def unstandardise_data(self):
        assert hasattr(self, "original_mean"), "This data is not standardised!"
        name = self.model_name + " unstandardised"
        unstandardised_data = self.model_data * self.original_std + self.original_mean
        unstandardised_model = ProcessModel(unstandardised_data, name)
        return unstandardised_model

    def calculate_anomaly(
        self,
        climatology_dates=["1961-01-01", "1990-12-31"],
        climatology=False,
        resample_freq=None,
    ):
        """Calculates the anomaly of a model relative to a specified climatological period (default 1961-1990 to match HadCRUT data).

        Args:
            climatology_dates (list, optional): Contains the start and end dates of the climatological period. Format must be parseable by xarray.sel e.g. ["YYYY-MM-DD", "YYYY-MM-DD"]. Defaults to ["1961-01-01", "1990-12-31"].
            climatology (xr.DataArray, optional): Use a precalculated climatology (e.g. ProcessModel.climatology). First dimension must be month. Defaults to False.
            resample_freq (str, optional): The temporal frequency to resample the data to if desired e.g. 'Y' for yearly. Defaults to None.

        Returns:
            ProcessModel: The anomalised model data
        """
        # If a climatology is not specified, it calculates one and returns it
        da = self.model_data.copy(deep=True)
        if np.any(climatology) == False:
            t0 = climatology_dates[0]
            t1 = climatology_dates[1]
            clim_years = da.sel(time=slice(t0, t1))
            clim = clim_years.groupby("time.month").mean().mean("realisation")
        else:
            clim = climatology
            assert (
                clim.month.size == 12
            ), "Climatology is the incorrect length (must be 12)"
        da_anom = da.groupby("time.month") - clim
        da_anom = da_anom.drop_vars("month")
        if resample_freq:
            da_anom = da_anom.resample(time=resample_freq).mean()
        # Save climatology
        anomaly_model = ProcessModel(da_anom, self.model_name + " anomaly")
        anomaly_model.climatology = clim

        return anomaly_model

    def plot(self, **kwargs) -> tp.Any:
        """Plot the model data including mean and individual realisations.
        If there are spatial dimensions, these are collapsed to a spatial mean for plotting.

        Returns:
            plt.ax: Current axis of the plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        x = self.model_data.time
        if self.model_data.ndim > 2:
            warnings.warn("Collapsing (mean) non-time dimensions for plotting")
            coord_names = [coord for coord in self.model_data.coords]
            coord_names.remove("time")
            coord_names.remove("realisation")
            da = self.model_data.mean(coord_names)
        else:
            da = self.model_data
        ax.set_prop_cycle(get_style_cycler())
        for real in da.realisation:
            ax.plot(
                x,
                da.sel(realisation=real),
                alpha=0.1,
                color="gray",
                label="Realisations",
                ls="-",
            )
        ax.plot(x, da.mean("realisation"), label="Model mean", alpha=0.7)
        ax.legend(loc="best")
        ax = _unique_legend(ax)
        ax.set_title(self.model_name)
        return ax

    @property
    def mean_across_realisations(self):
        """Calcualte the model mean across the realisation axis

        Returns:
            xr.DataArray: The model averaged over realisations
        """
        return self.model_data.mean("realisation")

    @property
    def std_across_realisations(self):
        """Calcualte the model standard deviation across the realisation axis

        Returns:
            xr.DataArray: The model standard deviation
        """
        return self.model_data.std("realisation")

    @property
    def ndim(self):
        """The number of dimensions described by the model e.g. (realisation, time, lat) = 3 dimensions

        Returns:
            int: The number of dimensions.
        """
        return self.model_data.ndim

    @property
    def distribution(self) -> Distribution:
        """Returns the model posterior

        Raises:
            NotImplementedError: Not currenlty implemented for > 2 dimensions (i.e. realisation and time)

        Returns:
            distrax.Distribution: The model posterior
        """
        return self._distribution

    @distribution.setter
    def distribution(self, dist: Distribution):
        self._distribution = dist

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
    """A data class to collect together multiple ProcessModels. For example this class
    could contain the surface temperature output for 10 different models.

    Args:
        model_data (list): A list of ProcessModels.

    Returns:
        ModelCollection: _description_
    """

    models: tp.List[ProcessModel]
    idx: int = 0

    def __post_init__(self):
        self.check_time_axes()

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

    def fit(self, model, **kwargs):
        """A function to fit a statistical model to the process models within the ModelCollection, to learn the models' posteriors

        Args:
            model (AbstractModel): A statistical model, e.g. MeanFieldApproximation
        """
        for process_model in self.models:
            if process_model.distribution != None:
                warnings.warn("Removing the model's previously learnt distribution")
            dist = model.fit(process_model, **kwargs)
            process_model.distribution = dist

    @property
    def time(self) -> ColumnVector:
        """Returns the time dimension which is useful for plotting and comparing to other models and observations. Note
        this doesn't work for plotting with plt.fill_between(...). For this purpose, use ProcessModel.time.values.

        Returns:
            xr.DataArray: A data array object of the time dimension
        """
        return self.models[0].time

    @property
    def max_val(self) -> int:
        """Returns the maximum value of the the ModelCollection.

        Returns:
            int: The maximum value
        """
        return np.max([model.max_val for model in self.models])

    @property
    def min_val(self) -> int:
        """Returns the minimum value of the the ModelCollection.

        Returns:
            int: The minimum value
        """
        return np.min([model.min_val for model in self.models])

    @property
    def number_of_models(self):
        """Returns the number of models held within the ModelCollection

        Returns:
            int: Number of models
        """
        return len(self.models)

    @property
    def model_names(self):
        """Returns a list of model names within the ModelCollection

        Returns:
            list: Model names
        """
        return [model.model_name for model in self.models]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, item):
        return self.models[item]

    def distributions(self) -> tp.Dict[str, Distribution]:
        """Returns a dictionary of model distributions (posteriors) where the keys
        to the dictionary are the model names

        Returns:
            tp.Dict[str, distrax.Distribution]: Dictionary of model distributions
        """
        return {model.model_name: model.distribution for model in self.models}

    def plot_all(
        self,
        ax: tp.Optional[tp.Any] = None,
        legend: bool = False,
        one_color: str = None,
        **kwargs
    ) -> tp.Any:
        """Plots all the models within the ModelCollection on one axes, without the individual realisations

        Args:
            ax (tp.Optional[plt.ax], optional): An axes to add the trace to. If not specified one is generated. Defaults to None.
            legend (bool, optional): Boolean to toggle displaying the legend. Defaults to False.
            one_color (str, optional): Color name e.g. 'tab:blue', specified if you want all the models to be plotted the same colour. Defaults to None.

        Returns:
            plt.ax: A matplotlib axis
        """
        if not ax:
            fig, ax = plt.subplots(figsize=(15, 7))

        ax.set_prop_cycle(get_style_cycler())
        for model in self:
            if model.ndim > 2:
                warnings.warn("Collapsing (mean) non-time dimensions for plotting")
                coord_names = [coord for coord in model.model_data.coords]
                coord_names.remove("time")
                da = model.model_data.mean(coord_names)
            else:
                da = model.model_data.mean("realisation")
            x = model.time

            if one_color:
                ax.plot(x, da.values, alpha=0.3, color=one_color)
            else:
                ax.plot(x, da.values, alpha=0.5, label=model.model_name)
        if legend:
            ax.legend(loc="best")
        return ax

    def plot_grid(self, **kwargs) -> tp.Any:
        """Plots all the models within the ModelCollection on seperate axes (1 per model), with the individual realisations"""
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
                warnings.warn("Collapsing (mean) non-time dimensions for plotting")
                coord_names = [coord for coord in model.model_data.coords]
                coord_names.remove("time")
                model_mean = model.model_data.mean(coord_names)
                coord_names.remove("realisation")
                reals = [real.mean(coord_names) for real in model]
            else:
                reals = [real for real in model]
                model_mean = model.mean_across_realisations

            ax.plot(x, model_mean, alpha=0.7, label=model.model_name, **args, zorder=10)

            [
                ax.plot(
                    x, real, alpha=0.1, color="gray", label="Realisations", zorder=1
                )
                for real in reals
            ]
            ax.legend(loc="best")
            ax = _unique_legend(ax)

        fig.show()

    # Don't think we use these anymore MA: 24/5/22
    # @property
    # def covariance_stack(self) -> jnp.DeviceArray:
    #     return jnp.stack([model.distribution.covariance() for model in self.models])

    # @property
    # def mean_stack(self) -> jnp.DeviceArray:
    #     return jnp.stack([model.distribution.mean() for model in self.models])

    def check_time_axes(self):
        """Helper function used when creating the ModelCollection to check that all model time coords are the same.
        This avoids slight mismatches in time where models might use different calendars. Also fixes common issues
        where the middle of the month might be represented either as the 15th or 16th day.
        """
        time_axes_match = True
        for model1 in self.models:
            for model2 in self.models:
                if np.any(
                    model1.model_data.time.values != model2.model_data.time.values
                ):
                    time_axes_match = False
        if time_axes_match == False:
            warnings.warn(
                "Time axes of models don't match: applying naive fix. Check models are collocated correctly in time!"
            )
            new_time = self.time
            for model in self:
                model.model_data["time"] = new_time

        return
