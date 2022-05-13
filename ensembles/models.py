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
import warnings

from .data import ModelCollection
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
