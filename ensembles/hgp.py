import tensorflow as tf
import numpy as np
import gpflow
from gpflow.config import default_jitter, default_float
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.mean_functions import Zero

import gpflow
from gpflow.models import BayesianModel

from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.mean_functions import Zero

from copy import deepcopy
from tqdm import trange
import typing as tp


class HSGP(BayesianModel, InternalDataTrainingLossMixin):
    # Bayesian model required for the ability to use training_loss as MLLO
    # InternalData... required as the gp model will hold it's own data.

    def __init__(
        self,
        data,
        group_kernel=None,
        individual_kernel=None,
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
        self.num_data = self.X.shape[0]
        # These deep copy statements could be problematic...
        # Group kernel
        self.K_group = deepcopy(group_kernel)
        # Inidividual kernels
        self.K_individual_list = [deepcopy(individual_kernel) for i in range(self.Y.shape[1])]

        # Default is zero mean function
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function

        # Should probably set noise variance as trainable(?)
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(noise_variance)

        self.elbos = []

    def maximum_log_likelihood_objective(self):
        # The InternalDataTrainingLossMixin provides the training_loss method.
        # And is for models that contain their data.
        #
        return self.total_elbo()

    # def total_elbo_map(self):

    #     elements = [] #X, Y, Kernel for each calculation
    #     for i in range(len(self.K_individual_list)):
    #         for j in range(len(self.K_individual_list)):
    #             if i==j:
    #                 elements.append([
    #                     self.X,
    #                     self.Y[i],
    #                     self.K_individual_list[i] + self.K_group])
    #             else:
    #                 elements.append([
    #                     self.X,
    #                     self.Y[i],
    #                     self.K_group])
    #     output = tf.reduce_sum(tf.map_fn(lambda x: calc_ELBO_part(*x), elements, fn_output_signature=tf.float64))

    #     return output

    # def sub_calc_ELBO_part(self, i, j):
    #     output = tf.constant(0.0, dtype=tf.float64)
    #     # This currently can't be decorated with tf.function because of the 
    #     # indexing of the list of kernels. I think tf doesn't know how to represent
    #     # the kernel objects on the graph. This happens because it is trying to 
    #     # tensorise the kernels. I considered using tf.switch_case but didn't think that would help
    #     if i != j:
    #         output = self.calc_ELBO_part(
    #             self.X,
    #             tf.expand_dims(self.Y[:, tf.constant(i)], -1),
    #             self.K_group)
    #     elif i == j:
    #         output = self.calc_ELBO_part(
    #             self.X,
    #             tf.expand_dims(self.Y[:, tf.constant(i)], -1),
    #             self.K_individual_list[i] + self.K_group
    #         )
    #     return output

    # def add_noise_cov(self, K: tf.Tensor, likelihood_variance) -> tf.Tensor:
    #     """
    #     Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
    #     and I is the corresponding identity matrix.
    #     """
    #     k_diag = tf.linalg.diag_part(K)
    #     s_diag = tf.fill(tf.shape(k_diag), likelihood_variance)
    #     return tf.linalg.set_diag(K, k_diag + s_diag)

    def predict_f(
        self, xtest: tp.Union[np.ndarray, tf.Tensor],
        kernel: gpflow.kernels.base.Kernel,
        Y: tp.Union[np.ndarray, tf.Tensor]
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:

        Xi = self.X
        # For group kernels average the Ys. Where Y is just (D x 1), 
        # the output will still be the same
        err = tf.reshape(tf.reduce_mean(self.Y, axis=1), (-1, 1)) - self.mean_function(Xi)
        
        num_inducing = self.inducing_points.num_inducing
        kuf = Kuf(self.inducing_points, kernel, Xi)
        kuu = Kuu(self.inducing_points, kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_points, kernel, xtest)

        sigma = tf.sqrt(self.likelihood.variance)
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
        self, xtest: tp.Union[np.ndarray, tf.Tensor]
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        
        f_mean, f_var = self.predict_f(
                            xtest,
                            self.K_group,
                            self.Y 
                        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_individual(
        self,
        xtest: tp.Union[np.ndarray, tf.Tensor],
        individual_idx: int
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        
        f_mean, f_var = self.predict_f(
                            xtest,
                            self.K_group + self.K_individual_list[individual_idx],
                            self.Y[:, individual_idx] 
                        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_individual_without_group(
        self,
        xtest: tp.Union[np.ndarray, tf.Tensor],
        individual_idx: int
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        
        f_mean, f_var = self.predict_f(
                            xtest,
                            self.K_individual_list[individual_idx],
                            self.Y[:, individual_idx] 
                        )
        return f_mean, f_var

    # def total_elbo_while(self):
    #     i = tf.constant(0, dtype=tf.int32)
    #     j = tf.constant(0, dtype=tf.int32)
    #     tot = tf.constant(0, dtype=tf.float64)

    #     # i_loop
    #     def i_body(i, j, tot):
    #         tot = tf.add(tot, self.sub_calc_ELBO_part(i, j))
    #         return tf.add(i, 1), j, tot

    #     # c_i = lambda i, j, tot: tf.less(i, j + 1)
    #     c_i = lambda i, j, tot: tf.less(i, len(self.K_individual_list))
    #     b_i = lambda i, j, tot: i_body(i, j, tot)
    #     # b_i = lambda i, j, tot: (tf.add(i, 1), j, tf.add(tot, self.sub_calc_ELBO_part(i, j)))

    #     def j_body(i, j, tot):
    #         tot = tf.while_loop(c_i, b_i, [0, j, tot])[2]
    #         return i, tf.add(j, 1), tot

    #     # j_loop
    #     c_j = lambda i, j, tot: tf.less(j, len(self.K_individual_list))
    #     b_j = lambda i, j, tot: j_body(0, j, tot)

    #     tot_fin = tf.while_loop(c_j, b_j, [i, j, tot])[2]

    #     return tot_fin

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
                        kernel=self.K_individual_list[i] + self.K_group)
                    self.loss_sum = tf.add(
                        self.loss_sum,
                        elbo_part)
                else:
                    # If off diagonal term use group kernel
                    elbo_part = self.calc_ELBO_part(
                        self.X,
                        tf.expand_dims(self.Y[:, i], -1),
                        kernel=self.K_group)
                    self.loss_sum = tf.add(
                        self.loss_sum,
                        elbo_part)

        return self.loss_sum

    def calc_ELBO_part(self, X, Y, kernel):
        tf.debugging.assert_shapes([(X, ("N", "D")), (Y, ("N", 1))])
        # Using the code and approach from gpflow.models.SGPRBase_deprecated.upper_bound
        Kdiag = kernel(X, full_cov=False)
        # print(Kdiag.shape)
        # kfu = kernel.K(X, self.inducing_points)
        kuu = kernel.K(self.inducing_points.Z, self.inducing_points.Z)  # might want jitter
        # print(kuu.shape)
        kuf = kernel.K(self.inducing_points.Z, X)
        # print(kuf.shape)
        I = tf.eye(tf.shape(kuu)[0], dtype=default_float())
        Luu = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(Luu, kuf, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = I + AAT / self.likelihood.variance
        LB = tf.linalg.cholesky(B)

        # Trace term
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(tf.square(A))
        corrected_noise = self.likelihood.variance + c
        const = -0.5 * self.num_data * tf.math.log(2 * np.pi * self.likelihood.variance)
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        err = Y - self.mean_function(X)
        LC = tf.linalg.cholesky(I + AAT / corrected_noise)
        v = tf.linalg.triangular_solve(LC, tf.linalg.matmul(A, err) / corrected_noise, lower=True)
        quad = -0.5 * tf.reduce_sum(tf.square(err)) / corrected_noise + 0.5 * tf.reduce_sum(
            tf.square(v)
        )
        return const + logdet + quad

    def fit(self, params):
        self.objective_evals = []
        opt = tf.optimizers.Adam(learning_rate=params["learning_rate"])
        objective = tf.function(self.training_loss)

        tr = trange(params["optim_nits"])
        for i in tr:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss = objective()
                self.objective_evals.append(loss.numpy())
            grads = tape.gradient(loss, self.trainable_variables)
            opt.apply_gradients(zip(grads, self.trainable_variables))
