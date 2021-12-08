import abc
import typing as tp

import gpflow
import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.models import GPR
from gpflow.models.model import GPModel
from gpflow.optimizers import Scipy
from tqdm import tqdm

from .array_types import ColumnVector, Matrix


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


class StochasticGP(GPFlowModel):
    pass
