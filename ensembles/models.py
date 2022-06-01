import abc
import typing as tp
# from ensembles.data import Distribution
from pytest import param

import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import distrax as dx
import jax.numpy as jnp
import jax
import optax as ox
import gpflow as gpf
import numpy as np
import warnings
from tensorflow_probability.substrates.jax import bijectors as tfb
from tslearn.barycenters import dtw_barycenter_averaging_subgradient
from .array_types import ColumnVector, Matrix, Vector
from ensembles import data as es_data # Trying to avoid circular imports
import copy
import xarray as xr 
import ensembles as es


class AbstractModel:
    def __init__(self, name: str = "Model") -> None:
        self.name = name
        self.model = None

    def fit(self, X: Matrix, y: ColumnVector, params: dict) -> None:
        X_transformed = self.transform_X(X, training=True)
        y_transformed = self.transform_y(y, training=True)
        tf.debugging.assert_shapes(
            [
                (X, ("N", "D")),
                (X_transformed, ("N", "D")),
                (y, ("N", "Q")),
                (y_transformed, ("N", "Q")),
            ]
        )
        self._fit(X_transformed, y_transformed, params)

    def predict(self, X: Matrix, params: dict) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        X_transformed = self.transform_X(X, training=False)
        tf.debugging.assert_shapes([(X, ("N", "D")), (X_transformed, ("N", "K"))])
        mu, sigma2 = self._predict(X, params)
        mu, sigma2 = self.untransform_outputs(mu, sigma2)
        tf.debugging.assert_shapes([(X, ("N", "D")), (mu, ("N", "K")), (sigma2, ("N", "K"))])
        return mu, sigma2

    def transform_X(self, X: Matrix, training: bool = True):
        return X

    def transform_y(self, y: ColumnVector, training: bool = True):
        return y

    def untransform_outputs(
        self, mu: ColumnVector, sigma2: ColumnVector
    ) -> tp.Tuple[ColumnVector, ColumnVector]:
        return mu, sigma2

    @abc.abstractmethod
    def _fit(self, X: Matrix, y: ColumnVector, params: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, X: Matrix, params: dict) -> None:
        raise NotImplementedError


class MeanFieldApproximation:
    def __init__(
        self,
        name="MeanFieldModel",
    ):
        self.name = name

    def step_fn(self, samples: tf.Tensor, negative: bool = False) -> None:
        obs = samples # Was transposed for the 1D example - TODO must check
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def step(params: dict):
            mu = params["mean"]
            sigma = params["variance"]
            dist = dx.Normal(mu, sigma)
            log_prob = jnp.sum(dist.log_prob(obs))
            return log_prob * constant

        return step

    def fit(
        self,
        model: AbstractModel,
        optimiser: ox.GradientTransformation = None,
        n_optim_nits: int = 500,
        compile_objective: bool = False,
    ) -> es_data.Distribution:
        if not optimiser:
            optimiser = ox.adam(learning_rate=0.01)
            warnings.warn("No optimiser specified, using Adam with learning rate 0.01")

        realisation_set = jnp.asarray(model.model_data.values).reshape(model.n_realisations, -1)
        mean = jnp.mean(realisation_set, axis=0)
        variance = jnp.var(realisation_set, axis=0)

        params = {"mean": mean, "variance": variance}

        objective_fn = self.step_fn(realisation_set, negative=True)
        if compile_objective:
            objective_fn = jax.jit(objective_fn)

        opt_state = optimiser.init(params)

        tr = trange(n_optim_nits)
        for i in tr:
            val, grads = jax.value_and_grad(objective_fn)(params)
            updates, opt_state = optimiser.update(grads, opt_state)
            params = ox.apply_updates(params, updates)
            if i % 100 == 0:
                tr.set_description(f"Objective: {val: .2f}")

        blank_array = xr.ones_like(model.model_data[0].drop('realisation')) * np.nan
        blank_array = blank_array.rename('blank')

        dist = es_data.Distribution(
            mu=mean, covariance=variance, dim_array=blank_array,
            dist_type=dx.Normal)
        return dist

    # def return_distribution(self, params):
        # return dx.MultivariateNormalDiag(params["mean"], params["variance"])


# class FullRankApproximation:
#     def __init__(
#         self,
#         name="FullRankModel",
#     ):
#         self.name = name

#     def step_fn(self, samples: tf.Tensor, negative: bool = False) -> None:
#         obs = samples.T
#         constant = jnp.array(-1.0) if negative else jnp.array(1.0)
#         bij = tfb.FillScaleTriL()

#         def step(params: dict):
#             mu = params["mean"]
#             Lvec = params["covariance"]
#             L = bij.forward(Lvec)
#             dist = dx.MultivariateNormalTri(mu, L)
#             log_prob = jnp.sum(dist.log_prob(obs))
#             return log_prob * constant

#         return step

#     def fit(
#         self,
#         model: AbstractModel,
#         optimiser: ox.GradientTransformation = None,
#         n_optim_nits: int = 500,
#         compile_objective: bool = False,
#     ) -> dx.Distribution:
#         if not optimiser:
#             optimiser = ox.adam(learning_rate=0.005)
#             warnings.warn("No optimiser specified, using Adam with learning rate 0.005")
#         if model.model_data.ndim > 2:
#              raise NotImplementedError('Not implemented for more than temporal dimensions')
#         realisation_set = jnp.asarray(model.model_data.values)
#         mean = jnp.mean(realisation_set, axis=0)
#         covariance = jnp.eye(mean.shape[0]).astype(jnp.float32)

#         bij = tfb.FillScaleTriL()
#         covariance = bij.inverse(covariance)

#         params = {"mean": mean, "covariance": covariance}
#         objective_fn = self.step_fn(realisation_set, negative=True)

#         if compile_objective:
#             objective_fn = jax.jit(objective_fn)

#         opt_state = optimiser.init(params)

#         tr = trange(n_optim_nits)
#         for i in tr:

#             val, grads = jax.value_and_grad(objective_fn)(params)
#             updates, opt_state = optimiser.update(grads, opt_state)
#             params = ox.apply_updates(params, updates)
#             if i % 100 == 0:
#                 tr.set_description(f"Objective: {val: .2f}")
#         return self.return_distribution(params)

#     def return_distribution(self, params):
#         bij = tfb.FillScaleTriL()
#         L = bij.forward(params["covariance"])
#         return dx.MultivariateNormalTri(params["mean"], L)


class _HeteroskedasticGaussian(gpf.likelihoods.Likelihood):
    def __init__(self, **kwargs):
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpf.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError


class GPDTW1D:
    def __init__(self, name: str = "GPRegressor") -> None:
        self.name = name

    def fit(
        self,
        model: AbstractModel,
        n_optim_nits: int = 500,
        compile_objective: bool = False,
    ) -> es_data.Distribution:
        if model.model_data.ndim > 2:
            raise NotImplementedError('Not implemented for more than temporal dimensions. Use GPDTW3D instead')
        realisation_set = model.model_data.values
        y_mean = dtw_barycenter_averaging_subgradient(realisation_set, max_iter=50, tol=1e-3)
        y_var = np.var(realisation_set, axis=0).reshape(-1, 1)
        Y = np.concatenate([y_mean, y_var], axis=1)

        X = realisation_set.T
        # X = np.linspace(0, 1, y_mean.shape[0]).reshape(-1, 1)

        likelihood = _HeteroskedasticGaussian()
        kernel = gpf.kernels.Matern32()
        gp_model = gpf.models.VGP((X, Y), kernel=kernel, likelihood=likelihood, num_latent_gps=1)

        natgrad = gpf.optimizers.NaturalGradient(gamma=0.5)
        adam = tf.optimizers.Adam(0.01)

        gpf.utilities.set_trainable(gp_model.q_mu, False)
        gpf.utilities.set_trainable(gp_model.q_sqrt, False)

        if compile_objective:
            loss = tf.function(gp_model.training_loss)
        else:
            loss = gp_model.training_loss

        tr = trange(n_optim_nits)
        losses = []

        for i in tr:
            natgrad.minimize(loss, [(gp_model.q_mu, gp_model.q_sqrt)])
            adam.minimize(loss, gp_model.trainable_variables)
            if i % 25 == 0:
                l = loss().numpy()
                tr.set_postfix({"loss": f"{l :.2f}"})
                losses.append(l)

        mu, cov = gp_model.predict_f(X, full_cov=True, full_output_cov=False)
        mu = mu.numpy().squeeze()
        cov = cov.numpy().squeeze()
        cov += np.diag(Y[:, 1])
        blank_array = xr.ones_like(model.model_data[0].drop('realisation')) * np.nan
        blank_array = blank_array.rename('blank')

        dist = es_data.Distribution(
            mu=mu, covariance=cov, dim_array=blank_array,
            dist_type=dx.MultivariateNormalFullCovariance)
        return dist


class GPDTW3D:
    def __init__(self, name: str = "GP3DRegressor") -> None:
        self.name = name

    def fit(
        self,
        model: AbstractModel,
        n_optim_nits: int = 500,
        compile_objective: bool = False,
    ) -> es_data.Distribution:
        if not model.model_data.ndim == 4:
            raise NotImplementedError('This method is only implemented for 4 dimensions (realisation, time, latitude, longitude')
        
        ## Data prep
        # Check coordinate names
        assert 'latitude' in model.model_data.coords, "There must be a latitude coordinate in the dataArray"
        assert 'longitude' in model.model_data.coords, "There must be a longitude coordinate in the dataArray"

        # Check latitude is third dim
        if np.where(np.asarray(model.model_data.dims) == 'latitude')[0][0] != 2:
            raise IndexError('Coordinate order should be realisation, time, latitude, longitude')

        lats = model.model_data.latitude
        lons = model.model_data.longitude

        fitted_mean = np.zeros_like(model.model_data.values[0])
        fitted_var = np.zeros_like(model.model_data.values[0])

        # Compute the DTW for every latitude and longitude
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                realisation_set = model.model_data.sel(latitude=lat, longitude=lon).values
                y_mean = dtw_barycenter_averaging_subgradient(realisation_set, max_iter=50, tol=1e-3).squeeze()
                y_var = np.var(realisation_set, axis=0)
                fitted_mean[:, i, j] = y_mean
                fitted_var[:, i, j] = y_var

        # Put DTW output and variance into xarray DataArrays for dimension management
        mean_array = copy.deepcopy(model.model_data.isel(realisation=0)).drop_vars('realisation')
        mean_array.data = fitted_mean
        var_array = copy.deepcopy(model.model_data.isel(realisation=0)).drop_vars('realisation')
        var_array.data = fitted_var

        # Extend coordinates from 2D -> 3D for ease of fitting GP. 
        lon_grid, lat_grid = np.meshgrid(mean_array.longitude, mean_array.latitude)
        x = np.cos(lat_grid * np.pi / 180)  * np.cos(lon_grid * np.pi / 180)
        y = np.cos(lat_grid * np.pi / 180)  * np.sin(lon_grid * np.pi / 180)
        z = np.sin(mean_array.latitude.values * np.pi / 180)

        # Add in more desciptive time coordinates
        t_cont = np.arange(len(mean_array.time))
        t_cont = 2 * t_cont / np.max(t_cont) - 1
        t_month = mean_array.time.dt.month.values
        t_sin = np.sin(2 * np.pi * t_month / 12)
        t_cos = np.cos(2 * np.pi * t_month / 12)

        # Add these auxillary coordinates to the DataArray
        mean_array = mean_array.assign_coords(
                        x=(["latitude", 'longitude'], x),
                        y=(["latitude", 'longitude'], y),
                        z=("latitude", z),
                        t_cont=("time", t_cont),
                        t_sin=("time", t_sin),
                        t_cos=("time", t_cos),
                        )

        # X includes (as a flattened array):
        #   - all of the above custom coords (x,y,z,t_cont,t_sin,t_cos)
        #   - all of the realisations 
        X = np.concatenate(
            [mean_array.to_dataframe().drop(mean_array.name, axis=1).values,
            model.model_data.to_dataframe().tas.values.reshape(len(model.model_data.realisation), -1).T],
            axis=1).astype(np.float64)

        # Y includes (as a flattened array):
        #   - The DTW fit
        #   - Variance
        Y = np.concatenate(
            [
                mean_array.to_dataframe()[mean_array.name].values.reshape(-1, 1),
                var_array.to_dataframe()[var_array.name].values.reshape(-1, 1)
            ],
            axis=1).astype(np.float64)

        # Fit a GP to this data
        likelihood = _HeteroskedasticGaussian()

        # 4 kernels (1 for x,y space, 1 for latitude (z), 1 for time, 1 for the realisations)
        time_kernel = gpf.kernels.Matern32(active_dims=[3, 4, 5])
        x_y_kernel = gpf.kernels.Matern32(active_dims=[0, 1])
        z_kernel = gpf.kernels.Matern32(active_dims=[2])
        realisation_kernel = gpf.kernels.Matern32(active_dims=list(np.arange(6, 6 + model.n_realisations)))
        kernel = time_kernel + x_y_kernel + z_kernel + realisation_kernel

        gp_model = gpf.models.GPMC((X, Y), kernel=kernel, likelihood=likelihood, num_latent_gps=1)

        adam = tf.optimizers.Adam(0.01)
        loss = tf.function(gp_model.training_loss)

        tr = tqdm.trange(n_optim_nits)
        losses = []

        # Fit GP in this loop
        for i in tr:
            adam.minimize(loss, gp_model.trainable_variables)
            if i % 1 == 0:
                l = loss().numpy()
                tr.set_postfix({"loss": f"{l :.2f}"})
                losses.append(l)
        
        # Get fit output from GP
        mu, cov = gp_model.predict_f(X, full_cov=True, full_output_cov=False)
        mu = mu.numpy().squeeze()
        cov = cov.numpy().squeeze()
        cov += np.diag(Y[:, 1])

        blank_array = xr.ones_like(model.model_data[0].drop('realisation')) * np.nan
        blank_array = blank_array.rename('blank')

        dist = es_data.Distribution(
            mu=mu, covariance=cov, dim_array=blank_array,
            dist_type=dx.MultivariateNormalFullCovariance)
        return dist


# class JointReconstruction(tf.Module):
#     def __init__(self, means: Vector, variances: Vector, name="JointReconstruction"):
#         super().__init__(name=name)
#         self.mu, self.sigma_hat = self._build_parameters(means, variances)
#         self.objective_evals = []

#     def fit(self, samples: tf.Tensor, params: dict, compile_fn: bool = False) -> None:
#         opt = tf.optimizers.Adam(learning_rate=params["learning_rate"])
#         if compile_fn:
#             objective = tf.function(self._objective_fn())
#         else:
#             objective = self._objective_fn()

#         for _ in trange(params["optim_nits"]):
#             with tf.GradientTape() as tape:
#                 tape.watch(self.trainable_variables)
#                 loss = objective(samples)
#                 self.objective_evals.append(loss.numpy())
#             grads = tape.gradient(loss, self.trainable_variables)
#             opt.apply_gradients(zip(grads, self.trainable_variables))

#     def return_parameters(self) -> tp.Tuple[tf.Tensor, tf.Tensor]:
#         return tf.convert_to_tensor(self.mu), tf.convert_to_tensor(self.sigma_hat)

#     def return_joint_distribution(self) -> tfp.distributions.Distribution:
#         return tfp.distributions.MultivariateNormalTriL(self.mu, self.sigma_hat)

#     def _objective_fn(self):
#         dist = tfp.distributions.MultivariateNormalTriL(self.mu, self.sigma_hat)

#         def log_likelihood(x: tf.Tensor) -> tf.Tensor:
#             return -tf.reduce_sum(dist.log_prob(x))

#         return log_likelihood

#     @staticmethod
#     def _build_parameters(
#         means, variances
#     ) -> tp.Tuple[tfp.util.TransformedVariable, tfp.util.TransformedVariable]:
#         mu = tfp.util.TransformedVariable(
#             initial_value=tf.cast(means, dtype=tf.float64),
#             bijector=tfp.bijectors.Identity(),
#             dtype=tf.float64,
#             trainable=False,
#         )
#         sigma_hat = tfp.util.TransformedVariable(
#             initial_value=tf.cast(tf.eye(num_rows=means.shape[0]) * variances, dtype=tf.float64),
#             bijector=tfp.bijectors.FillTriangular(),
#             dtype=tf.float64,
#             trainable=True,
#         )
#         return mu, sigma_hat
