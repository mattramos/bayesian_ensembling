import abc
import typing as tp

import gpflow
from numpy.core.arrayprint import _leading_trailing
from numpy.core.fromnumeric import var
import tensorflow as tf
from gpflow import inducing_variables
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.models import GPR, SGPR
from gpflow.models.model import GPModel
from gpflow.optimizers import Scipy
from scipy.cluster.vq import kmeans2
from tensorflow.python.ops.gen_math_ops import Mod
import tensorflow_probability as tfp
from tqdm import tqdm, trange

from .array_types import ColumnVector, Matrix, Vector


class Model:
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


class GPFlowModel(Model):
    def __init__(self, kernel: Kernel, name: str = "GPFlow Model") -> None:
        super().__init__(name=name)
        self.kernel = kernel
        self.model: GPModel = None

    def _predict(self, X: Matrix, params: dict) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        tf_data = tf.data.Dataset.from_tensor_slices(X).batch(params["batch_size"])

        @tf.function
        def pred(batch):
            return self.model.predict_y(batch)

        pred_matrix = tf.concat(
            [
                b
                for b in tqdm(
                    tf_data.prefetch(tf.data.AUTOTUNE).map(
                        pred, num_parallel_calls=tf.data.AUTOTUNE
                    ),
                    desc="Predicting",
                )
            ],
            axis=1,
        )
        return pred_matrix[0, :, :], pred_matrix[1, :, :]

    @abc.abstractmethod
    def _fit(self, X: Matrix, y: ColumnVector, params: dict) -> None:
        return super()._fit(X, y, params)


class ConjugateGP(GPFlowModel):
    def __init__(self, kernel: Kernel, name: str = "GPR") -> None:
        super().__init__(kernel, name=name)

    def _fit(self, X: Matrix, y: ColumnVector, params: dict):
        self.model = GPR((X, y), kernel=self.kernel)
        nits = params["optim_nits"]
        Scipy().minimize(
            self.model.training_loss,
            self.model.trainable_variables,
            options=dict(maxiter=nits),
        )


class SparseGP(GPFlowModel):
    def __init__(self, kernel: Kernel, name: str = "Sparse GPR") -> None:
        super().__init__(kernel, name=name)
        self.Z = None
        self.elbos = []

    def _fit(self, X: Matrix, y: ColumnVector, params: dict) -> None:
        # Define model
        Z = self._constuct_inducing_locs(X, params["n_inducing"])
        self.Z = Z
        self.model = SGPR((X, y), kernel=self.kernel, inducing_variable=Z)

        # Define optimiser
        opt = tf.optimizers.Adam(learning_rate=params["learning_rate"])

        # Compile loss fn.
        @tf.function
        def step():
            opt.minimize(self.model.training_loss, self.model.trainable_variables)

        # Run optimisation
        tr = trange(params["optim_nits"])
        for nit in tr:
            step()
            if nit % params["log_interval"] == 0:
                elbo = -self.model.training_loss().numpy()
                self.elbos.append(elbo)
                tr.set_postfix({"ELBO": elbo})

    def _constuct_inducing_locs(self, X: Matrix, n_inducing: int) -> InducingPoints:
        Z = kmeans2(data=X, k=n_inducing, minit="points")[0]
        return InducingPoints(Z)


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
