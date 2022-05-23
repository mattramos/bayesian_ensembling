import abc
import typing as tp
from pytest import param

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
        obs = samples.T
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def step(params: dict):
            mu = params["mean"]
            sigma = params["variance"]
            dist = dx.MultivariateNormalDiag(mu, sigma)
            log_prob = jnp.sum(dist.log_prob(obs))
            return log_prob * constant

        return step

    def fit(
        self,
        model: AbstractModel,
        optimiser: ox.GradientTransformation = None,
        n_optim_nits: int = 500,
        compile_objective: bool = False,
    ) -> dx.Distribution:
        if not optimiser:
            optimiser = ox.adam(learning_rate=0.01)
            warnings.warn("No optimiser specified, using Adam with learning rate 0.01")
        realisation_set = jnp.asarray(model.model_data.values)
        mean = jnp.mean(realisation_set, axis=1)
        variance = jnp.var(realisation_set, axis=1)

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
        return self.return_distribution(params)

    def return_distribution(self, params):
        return dx.MultivariateNormalDiag(params["mean"], params["variance"])


class FullRankApproximation:
    def __init__(
        self,
        name="MeanFieldModel",
    ):
        self.name = name

    def step_fn(self, samples: tf.Tensor, negative: bool = False) -> None:
        obs = samples.T
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        bij = tfb.FillScaleTriL()

        def step(params: dict):
            mu = params["mean"]
            Lvec = params["covariance"]
            L = bij.forward(Lvec)
            dist = dx.MultivariateNormalTri(mu, L)
            log_prob = jnp.sum(dist.log_prob(obs))
            return log_prob * constant

        return step

    def fit(
        self,
        model: AbstractModel,
        optimiser: ox.GradientTransformation = None,
        n_optim_nits: int = 500,
        compile_objective: bool = False,
    ) -> dx.Distribution:
        if not optimiser:
            optimiser = ox.adam(learning_rate=0.005)
            warnings.warn("No optimiser specified, using Adam with learning rate 0.005")
        realisation_set = jnp.asarray(model.model_data.values)
        mean = jnp.mean(realisation_set, axis=1)

        covariance = jnp.eye(mean.shape[0]).astype(jnp.float32)

        bij = tfb.FillScaleTriL()
        covariance = bij.inverse(covariance)

        params = {"mean": mean, "covariance": covariance}
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
        return self.return_distribution(params)

    def return_distribution(self, params):
        bij = tfb.FillScaleTriL()
        L = bij.forward(params["covariance"])
        return dx.MultivariateNormalTri(params["mean"], L)


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


class GPDTW:
    def __init__(self, name: str = "GPRegressor") -> None:
        self.name = name

    def fit(
        self,
        model: AbstractModel,
        n_optim_nits: int = 500,
        compile_objective: bool = False,
    ) -> dx.Distribution:
        realisation_set = model.model_data.values
        y_mean = dtw_barycenter_averaging_subgradient(realisation_set.T, max_iter=50, tol=1e-3)
        y_var = np.var(realisation_set, axis=1).reshape(-1, 1)
        Y = np.concatenate([y_mean, y_var], axis=1)

        X = realisation_set
        # X = np.linspace(0, 1, y_mean.shape[0]).reshape(-1, 1)

        likelihood = _HeteroskedasticGaussian()
        kernel = gpf.kernels.Matern32()
        model = gpf.models.VGP((X, Y), kernel=kernel, likelihood=likelihood, num_latent_gps=1)

        natgrad = gpf.optimizers.NaturalGradient(gamma=0.5)
        adam = tf.optimizers.Adam(0.01)

        gpf.utilities.set_trainable(model.q_mu, False)
        gpf.utilities.set_trainable(model.q_sqrt, False)

        if compile_objective:
            loss = tf.function(model.training_loss)
        else:
            loss = model.training_loss

        tr = trange(n_optim_nits)
        losses = []

        for i in tr:
            natgrad.minimize(loss, [(model.q_mu, model.q_sqrt)])
            adam.minimize(loss, model.trainable_variables)
            if i % 25 == 0:
                l = loss().numpy()
                tr.set_postfix({"loss": f"{l :.2f}"})
                losses.append(l)

        mu, sigma = model.predict_f(X, full_cov=True, full_output_cov=False)
        mu = mu.numpy().squeeze()
        sigma = sigma.numpy().squeeze()
        sigma += np.diag(Y[:, 1])
        dist = dx.MultivariateNormalFullCovariance(mu, sigma)
        return dist


class JointReconstruction(tf.Module):
    def __init__(self, means: Vector, variances: Vector, name="JointReconstruction"):
        super().__init__(name=name)
        self.mu, self.sigma_hat = self._build_parameters(means, variances)
        self.objective_evals = []

    def fit(self, samples: tf.Tensor, params: dict, compile_fn: bool = False) -> None:
        opt = tf.optimizers.Adam(learning_rate=params["learning_rate"])
        if compile_fn:
            objective = tf.function(self._objective_fn())
        else:
            objective = self._objective_fn()

        for _ in trange(params["optim_nits"]):
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss = objective(samples)
                self.objective_evals.append(loss.numpy())
            grads = tape.gradient(loss, self.trainable_variables)
            opt.apply_gradients(zip(grads, self.trainable_variables))

    def return_parameters(self) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        return tf.convert_to_tensor(self.mu), tf.convert_to_tensor(self.sigma_hat)

    def return_joint_distribution(self) -> tfp.distributions.Distribution:
        return tfp.distributions.MultivariateNormalTriL(self.mu, self.sigma_hat)

    def _objective_fn(self):
        dist = tfp.distributions.MultivariateNormalTriL(self.mu, self.sigma_hat)

        def log_likelihood(x: tf.Tensor) -> tf.Tensor:
            return -tf.reduce_sum(dist.log_prob(x))

        return log_likelihood

    @staticmethod
    def _build_parameters(
        means, variances
    ) -> tp.Tuple[tfp.util.TransformedVariable, tfp.util.TransformedVariable]:
        mu = tfp.util.TransformedVariable(
            initial_value=tf.cast(means, dtype=tf.float64),
            bijector=tfp.bijectors.Identity(),
            dtype=tf.float64,
            trainable=False,
        )
        sigma_hat = tfp.util.TransformedVariable(
            initial_value=tf.cast(tf.eye(num_rows=means.shape[0]) * variances, dtype=tf.float64),
            bijector=tfp.bijectors.FillTriangular(),
            dtype=tf.float64,
            trainable=True,
        )
        return mu, sigma_hat
