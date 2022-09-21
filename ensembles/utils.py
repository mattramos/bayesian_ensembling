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
import matplotlib.pyplot as plt
import os
import distrax
import pickle as pkl

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
            ensemble_method: AbstractEnsembleScheme,
            ssp: str,
            save_dir: str = None,
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
        self.ssp = ssp
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_fig_dir = os.path.join(save_dir, 'figs')
        if not os.path.exists(self.save_fig_dir):
            os.makedirs(self.save_fig_dir)
        self.save_csv_dir = os.path.join(save_dir, 'csvs')
        if not os.path.exists(self.save_csv_dir):
            os.makedirs(self.save_csv_dir)
        



    def _run_single_test(
            self,
            hindcast_models: ModelCollection,
            forecast_models: ModelCollection,
            pseudo_observations_past: ProcessModel,
            pseudo_observations_future: ProcessModel,
            n_optim_nits: int = 1000,
            use_prefit_models: bool = False):
        """A helper function to run a single iteration of the perfect model test, including validation against set metrics (NLL, RMSE, W2).

        Args:
            hindcast_models (ModelCollection): A ModelCollection of hindcast models.
            forecast_models (ModelCollection): A ModelCollection of forecast models. Must contain the same models as hindcast_models.
            pseudo_observations_past (ProcessModel): The pseudo observation used as past data.
            pseudo_observations_future (ProcessModel): The pseudo observation used as future data.
        """

        # Run the setup
        # Use prefit models if specified
        if use_prefit_models != True:
            hindcast_models.fit(model=self.emulate_method(), compile_objective=True, n_optim_nits=n_optim_nits, progress_bar=False)
            forecast_models.fit(model=self.emulate_method(), compile_objective=True, n_optim_nits=n_optim_nits, progress_bar=False)
            dist = self.emulate_method().fit(pseudo_observations_future, compile_objective=True, n_optim_nits=n_optim_nits)
            pseudo_observations_future.distribution = dist

        weight_function = self.weight_method()
        weights = weight_function(hindcast_models, pseudo_observations_past)
        # TODO: Add some functionality for constructing unpacked weights
        mean_weights = weights.mean('time')
        plt.figure()
        plt.bar(forecast_models.model_names, mean_weights.values)
        plt.ylabel('Weights')
        plt.xticks(rotation='vertical')
        save_path = os.path.join(self.save_fig_dir, f"weights/{weight_function.name}_with_{pseudo_observations_future.model_name}_as_pseudo_truth_{self.ssp}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        weights_single = weights.mean('time').expand_dims(time=forecast_models[0].model_data.time, axis=1)
        ensemble_method = self.ensemble_method()
        barycentre = ensemble_method(forecast_models, weights_single)

        # Validate it against set metrics for bary
        # Average NLL across realisations
        nll_bary = - jnp.mean(barycentre._dist.log_prob(pseudo_observations_future.model_data.values))
        # Ths is the average RMSE across realisations
        rmse_bary = np.mean(np.sqrt(np.mean((barycentre.mean - pseudo_observations_future.model_data)**2, axis=0)).values)
        # #Wasserstein distance
        if hasattr(pseudo_observations_future.distribution._dist, 'covariance'): # Check if the distribution has variance or covariance
            w2_bary = gaussian_w2_distance_distrax(barycentre._dist, pseudo_observations_future.distribution._dist, full_cov=True)
        else:
            w2_bary = gaussian_w2_distance_distrax(barycentre._dist, pseudo_observations_future.distribution._dist, full_cov=False)

        # Calculate MMM and metrics for the ensemble
        realisations = np.vstack([forecast_models[i].model_data.values for i in range(forecast_models.number_of_models)])
        mmm_dist = distrax.Normal(np.mean(realisations, axis=0), np.var(realisations, axis=0))
        nll_mmm = - jnp.mean(mmm_dist.log_prob(pseudo_observations_future.model_data.values))
        # Ths is the average RMSE across realisations
        rmse_mmm = np.mean(np.sqrt(np.mean((mmm_dist.mean() - pseudo_observations_future.model_data)**2, axis=0)).values)
        # #Wasserstein distance
        w2_mmm = gaussian_w2_distance_distrax(mmm_dist, pseudo_observations_future.distribution._dist, full_cov=False)


        from ensembles.plotters import cmap
        def plot_dist(dist, color='tab:blue', label='None', alpha=0.2, order=3):
            plt.plot(dist.mean.time, dist.mean, color=color, label=label, zorder=order)
            plt.fill_between(dist.mean.time.values, dist.mean - 2 * np.sqrt(dist.variance), dist.mean + 2 * np.sqrt(dist.variance), alpha=alpha, color=color, zorder=order-1, linewidth=0)

        plt.figure(figsize=(6.5, 4))
        plot_dist(barycentre, color=cmap()[0], label='Barycentre')
        plot_dist(pseudo_observations_future.distribution, color=cmap()[1], label='True model')
        plt.plot(barycentre.mean.time, mmm_dist.mean(), color=cmap()[2], label='MMM', zorder=3)
        plt.fill_between(barycentre.mean.time.values, mmm_dist.mean() - 2 * np.sqrt(mmm_dist.variance()), mmm_dist.mean() + 2 * np.sqrt(mmm_dist.variance()), alpha=0.2, color=cmap()[2], zorder=2, linewidth=0)
        # TODO - could generalise labels to be non temp related
        plt.xlabel('Time')
        plt.ylabel('Temperature anomally (°C) \n realitve to (1961-1990)')
        plt.legend()

        save_path = os.path.join(self.save_fig_dir, f"projs/{pseudo_observations_future.model_name}_as_pseudo_truth_{weight_function.name}_{self.ssp}.png")
        plt.savefig(save_path)
        plt.close()

        return nll_bary, rmse_bary, w2_bary, nll_mmm, rmse_mmm, w2_mmm


    def run(self, n_optim_nits: int = 1000, use_prefit_models=False):
        """Runs the perfect model test.

        Args:
            n_optim_nits (int, optional): The number of optimisation iterations to use. Defaults to 1000.
            save_file (bool, optional): Path and file name to save results to. Defaults to False (not saving).
        """

        df = pd.DataFrame(columns=['model as psuedo obs', f'nll_bary_{self.weight_method().name}', f'rmse_bary_{self.weight_method().name}', f'w2_bary_{self.weight_method().name}', 'nll_mmm', 'rmse_mmm', 'w2_mmm'])
        n_models = self.hindcast_models.number_of_models
        for i in range(n_models):
            # Copy hindcast models and remove the pseudo observations
            hindcast_model_list = copy.deepcopy(self.hindcast_models.models)
            pseudo_observations_past = hindcast_model_list.pop(i)
            # Same for the forecast models
            forecast_model_list = copy.deepcopy(self.forecast_models.models)
            pseudo_observations_future = forecast_model_list.pop(i)
            # Run the test
            nll_bary, rmse_bary, w2_bary, nll_mmm, rmse_mmm, w2_mmm = self._run_single_test(
                ModelCollection(hindcast_model_list),
                ModelCollection(forecast_model_list),
                pseudo_observations_past,
                pseudo_observations_future,
                n_optim_nits,
                use_prefit_models=use_prefit_models)
            df.loc[len(df.index)] = [
                pseudo_observations_past.model_name,
                nll_bary,
                rmse_bary,
                w2_bary,
                nll_mmm,
                rmse_mmm,
                w2_mmm
            ]
            # print(f'With {pseudo_observations_past.model_name} as pseudo obs: NLL: {nll}, RMSE: {rmse}, W2: {w2}')
        save_file = os.path.join(self.save_csv_dir, f'/prefect_model_test_results_{self.weight_method().name}_{self.ssp}.csv')
        df.to_csv(save_file)
        print(f'Saved results to {save_file}')