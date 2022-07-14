from ast import Return
from parso import parse
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy
import gpflow
import typing as tp
from argparse import ArgumentParser
import pandas as pd
from .data import ModelCollection, ProcessModel
from .models import AbstractModel
from .weights import AbstractWeight
from .ensemble_scheme import AbstractEnsembleScheme
import jax.numpy as jnp
from ensembles.wasserstein import gaussian_w2_distance_distrax

class PerfectModelTest:
    """Performs a perfect model test on the ensemble of models. This 
    is where a model is removed from the ensemble and used as 'psudo observations' 
    to test the ensembling framework. This way we can test the projective capabilities.
    """
    # Probably only works in 1D currently

    def __init__(
            self,
            hindcast_models: ModelCollection,
            forecast_models: ModelCollection,
            emulate_method: AbstractModel,
            weight_method: AbstractWeight,
            ensemble_method: AbstractEnsembleScheme
            ):
        """Initialisation of the perfect model test.

        Args:
            hindcast_models (ModelCollection): A ModelCollection of hindcast models.
            forecast_models (ModelCollection): A ModelCollection of forecast models. Must contain the same models as hindcast_models.
            emulate_method (AbstractModel): The method used to emulate the models e.g. ensembles.models.GPDTW1D
            weight_method (AbstractWeight):The weighting method used to weight the models e.g. ensembles.weights.LogLikelihoodWeight
            ensemble_method (AbstractEnsembleScheme): The method used to ensemble the model posteriors e.g. ensembles.ensemble_schemes.Barycentre
        """
        self.hindcast_models = hindcast_models
        self.forecast_models = forecast_models
        self.emulate_method = emulate_method
        self.weight_method = weight_method
        self.ensemble_method = ensemble_method

    def _run_single_test(
            self,
            hindcast_models: ModelCollection,
            forecast_models: ModelCollection,
            pseudo_observations_past: ProcessModel,
            pseudo_observations_future: ProcessModel,
            n_optim_nits: int = 1000):
        """A helper function to run a single iteration of the perfect model test, including validation against set metrics (NLL, RMSE, W2).

        Args:
            hindcast_models (ModelCollection): A ModelCollection of hindcast models.
            forecast_models (ModelCollection): A ModelCollection of forecast models. Must contain the same models as hindcast_models.
            pseudo_observations_past (ProcessModel): The pseudo observation used as past data.
            pseudo_observations_future (ProcessModel): The pseudo observation used as future data.
        """

        # Run the setup
        hindcast_models.fit(model=self.emulate_method(), compile_objective=True, n_optim_nits=n_optim_nits)
        forecast_models.fit(model=self.emulate_method(), compile_objective=True, n_optim_nits=n_optim_nits)
        dist = self.emulate_method().fit(pseudo_observations_future, compile_objective=True, n_optim_nits=n_optim_nits)
        pseudo_observations_future.distribution = dist
        weight_function = self.weight_method()
        weights = weight_function(hindcast_models, pseudo_observations_past)
        # TODO: Add some functionality for constructing unpacked weights
        weights_single = weights.mean('time').expand_dims(time=forecast_models[0].model_data.time, axis=1)
        ensemble_method = self.ensemble_method()
        barycentre = ensemble_method(forecast_models, weights_single)

        # Validate it against set metrics
        # Average NLL across realisations
        nll = - jnp.mean(barycentre._dist.log_prob(pseudo_observations_future.model_data.values))
        # Ths is the average RMSE across realisations
        rmse = np.mean(np.sqrt(np.mean((barycentre.mean - pseudo_observations_future.model_data)**2, axis=0)).values)
        # #Wasserstein distance
        if hasattr(pseudo_observations_future.distribution._dist, 'covariance'): # Check if the distribution has variance or covariance
            w2 = gaussian_w2_distance_distrax(barycentre._dist, pseudo_observations_future.distribution._dist, full_cov=True)
        else:
            w2 = gaussian_w2_distance_distrax(barycentre._dist, pseudo_observations_future.distribution._dist, full_cov=False)

        return nll, rmse, w2


    def run(self, n_optim_nits: int = 1000, save_file=False):
        """Runs the perfect model test.

        Args:
            n_optim_nits (int, optional): The number of optimisation iterations to use. Defaults to 1000.
            save_file (bool, optional): Path and file name to save results to. Defaults to False (not saving).
        """

        df = pd.DataFrame(columns=['model as psuedo obs', 'nll', 'rmse', 'w2'])
        n_models = self.hindcast_models.number_of_models
        for i in range(n_models):
            # Copy hindcast models and remove the pseudo observations
            hindcast_model_list = copy.deepcopy(self.hindcast_models.models)
            pseudo_observations_past = hindcast_model_list.pop(i)
            # Same for the forecast models
            forecast_model_list = copy.deepcopy(self.forecast_models.models)
            pseudo_observations_future = forecast_model_list.pop(i)
            # Run the test
            nll, rmse, w2 = self._run_single_test(
                ModelCollection(hindcast_model_list),
                ModelCollection(forecast_model_list),
                pseudo_observations_past,
                pseudo_observations_future,
                n_optim_nits)
            df.loc[len(df.index)] = [
                pseudo_observations_past.model_name,
                nll,
                rmse,
                w2
            ]
            print(f'With {pseudo_observations_past.model_name} as pseudo obs: NLL: {nll}, RMSE: {rmse}, W2: {w2}')

        if save_file:
            df.to_csv(save_file)
        else:
            return df




def simulate_data(
    n_obs: int,
    n_realisations: int,
    noise_lims: tp.Tuple[float, float],
    true_kernel: gpflow.kernels.Kernel,
    xlims: tp.Tuple[float, float] = (-5.0, 5.0),
    jitter_amount: float = 1e-8,
    seed_value: int = 123,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    tfp_seed = tfp.random.sanitize_seed(seed_value)
    rng = np.random.RandomState(123)
    X = np.sort(rng.uniform(*xlims, (n_obs, 1)), axis=0)
    true_kernel = gpflow.kernels.Matern32()
    Kxx = true_kernel(X) + tf.cast(tf.eye(n_obs) * jitter_amount, dtype=tf.float64)
    latent_y = tfp.distributions.MultivariateNormalTriL(
        np.zeros(n_obs), tf.linalg.cholesky(Kxx)
    ).sample(seed=tfp_seed)

    noise_terms = np.random.uniform(*noise_lims, size=n_realisations)
    realisations = []

    for noise in noise_terms:
        sample_y = latent_y.numpy() + rng.normal(
            loc=0.0, scale=noise, size=latent_y.numpy().shape
        )
        realisations.append(sample_y)
    Y = np.asarray(realisations).T
    return X, latent_y, Y
