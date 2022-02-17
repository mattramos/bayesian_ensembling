import tensorflow as tf
import numpy as np
import gpflow
from gpflow.config import default_jitter, default_float
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.mean_functions import Zero

import gpflow
from gpflow.models import BayesianModel
from gpflow.kernels import Kernel
from gpflow.inducing_variables import InducingPoints
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.mean_functions import Zero, MeanFunction

from copy import deepcopy
from tqdm import trange
import typing as tp


class HSGP(BayesianModel, InternalDataTrainingLossMixin):
    # Bayesian model required for the ability to use training_loss as MLLO
    # InternalData... required as the gp model will hold it's own data.

    def __init__(
        self,
        data,
        group_kernel,
        individual_kernel,
        noise_variance=1.0,
        mean_function=None,
        inducing_points=np.linspace(-3, 3, 50).reshape(-1, 1),
        name="HierarchicalSparseGP",
    ):
        super().__init__()
        # Tensorise data
        self.X, self.Y = data
        self.X = data_input_to_tensor(self.X)  # n_data x 1
        self.Y = data_input_to_tensor(self.Y)  # n_data x n_reals
        self.noise = 0.01
        self.inducing_points = inducingpoint_wrapper(inducing_points)
        self.num_inducing = inducing_points.shape[0]
        self.num_data = self.X.shape[0]
        # These deep copy statements could be problematic...
        # Group kernel
        self.K_group = deepcopy(group_kernel)
        # Inidividual kernels
        self.K_individual_list = [deepcopy(individual_kernel) for i in range(self.Y.shape[1])]
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(noise_variance)
        gpflow.utilities.set_trainable(self.likelihood.variance, False)
        self.group_likelihood = deepcopy(self.likelihood)
        self.individual_likelihoods = [deepcopy(self.likelihood) for i in range(self.Y.shape[1])]

        

        # Default is zero mean function
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function

        self.likelihoodvars = []
        self.consts = []
        self.logdets = []
        self.quads = []
        self.elbos = []

    def maximum_log_likelihood_objective(self):
        # The InternalDataTrainingLossMixin provides the training_loss method.
        # And is for models that contain their data.
        #
        return self.total_elbo()

    def predict_f(
        self,
        xtest: tp.Union[np.ndarray, tf.Tensor],
        kernel: gpflow.kernels.base.Kernel,
        Y: tp.Union[np.ndarray, tf.Tensor],
        likelihood: gpflow.likelihoods.base
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:

        Xi = self.X
        # For group kernels average the Ys. Where Y is just (D x 1),
        # the output will still be the same
        err = tf.reshape(tf.reduce_mean(self.Y, axis=1), (-1, 1)) - self.mean_function(Xi)

        num_inducing = self.inducing_points.num_inducing
        kuf = Kuf(self.inducing_points, kernel, Xi)
        kuu = Kuu(self.inducing_points, kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_points, kernel, xtest)

        sigma = tf.sqrt(likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        f_mean = mean + self.mean_function(xtest)
        var = (
            kernel(xtest, full_cov=False)
            + tf.reduce_sum(tf.square(tmp2), 0)
            - tf.reduce_sum(tf.square(tmp1), 0)
        )
        f_var = tf.tile(var[:, None], [1, 1])

        return f_mean, f_var

    def predict_group(
        self, xtest: tp.Union[np.ndarray, tf.Tensor],
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:

        f_mean, f_var = self.predict_f(xtest, self.K_group, self.Y, self.group_likelihood)
        return self.group_likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_individual(
        self, xtest: tp.Union[np.ndarray, tf.Tensor], individual_idx: int
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:

        f_mean, f_var = self.predict_f(
            xtest, self.K_group + self.K_individual_list[individual_idx], self.Y[:, individual_idx], self.individual_likelihoods[individual_idx]
        )
        return self.individual_likelihoods[individual_idx].predict_mean_and_var(f_mean, f_var)

    def predict_individual_without_group(
        self, xtest: tp.Union[np.ndarray, tf.Tensor], individual_idx: int
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:

        f_mean, f_var = self.predict_f(
            xtest, self.K_individual_list[individual_idx], self.Y[:, individual_idx], self.individual_likelihoods[individual_idx]
        )
        return self.individual_likelihoods[individual_idx].predict_mean_and_var(f_mean, f_var)

    def total_elbo(self):
        self.loss_sum = tf.constant(0.0, dtype=tf.float64)
        # Calculate elbo for all terms
        for i in range(len(self.K_individual_list)):
            for j in range(len(self.K_individual_list)):
                if i == j:
                    # If diagonal term add group and individual kernels
                    elbo_part = self.calc_ELBO_part(
                        self.X,
                        tf.expand_dims(self.Y[:, i], -1),
                        self.K_individual_list[i] + self.K_group,
                        self.individual_likelihoods[i]
                    )
                    self.loss_sum = tf.add(self.loss_sum, elbo_part)
                else:
                    # If off diagonal term use group kernel
                    elbo_part = self.calc_ELBO_part(
                        self.X, tf.expand_dims(self.Y[:, i], -1), self.K_group, self.group_likelihood
                    )
                    self.loss_sum = tf.add(self.loss_sum, elbo_part)

        return self.loss_sum

    def calc_ELBO_part(self, X, Y, kernel, likelihood):
        tf.debugging.assert_shapes([(X, ("N", "D")), (Y, ("N", 1))])
        # Using the code and approach from gpflow.models.SGPRBase_deprecated.upper_bound
        Kdiag = kernel(X, full_cov=False)
        # print(Kdiag.shape)
        # kfu = kernel.K(X, self.inducing_points)
        kuu = kernel.K(self.inducing_points.Z, self.inducing_points.Z)  # might want jitter
        kuu += tf.eye(self.num_inducing, dtype=tf.float64) * 1e-6
        kuf = kernel.K(self.inducing_points.Z, X)
        # print(kuf.shape)
        I = tf.eye(tf.shape(kuu)[0], dtype=default_float())
        Luu = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(Luu, kuf, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = I + AAT / likelihood.variance
        LB = tf.linalg.cholesky(B)

        # Trace term
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(tf.square(A))
        corrected_noise = likelihood.variance + c
        const = - 0.5 * self.num_data * tf.math.log(2 * np.pi * likelihood.variance)
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        err = Y - self.mean_function(X)
        LC = tf.linalg.cholesky(I + AAT / corrected_noise)
        v = tf.linalg.triangular_solve(LC, tf.linalg.matmul(A, err) / corrected_noise, lower=True)
        quad = -0.5 * tf.reduce_sum(tf.square(err)) / corrected_noise + 0.5 * tf.reduce_sum(
            tf.square(v)
        )
        self.quads.append(quad.numpy())
        self.logdets.append(logdet.numpy())
        self.consts.append(const.numpy())
        self.likelihoodvars.append(likelihood.variance.numpy())

        return const + logdet + quad

    def fit(self, params, compile: bool = False):
        self.objective_evals = []
        opt = tf.optimizers.Adam(learning_rate=params["learning_rate"])
        if compile:
            objective = tf.function(self.training_loss)
        else:
            objective = self.training_loss

        tr = trange(params["optim_nits"])
        for i in tr:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss = objective()
                self.objective_evals.append(loss.numpy())
            grads = tape.gradient(loss, self.trainable_variables)
            opt.apply_gradients(zip(grads, self.trainable_variables))
            tr.set_postfix({"ELBO": -loss.numpy()})

    def chol_solve(L, y):
        lx = solve_triangular(L, y, lower=True)
        x = solve_triangular(L, lx, lower=True, trans=1)
        return x