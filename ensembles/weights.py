import jax.numpy as jnp
from .data import ModelCollection, ProcessModel
import abc
from tqdm import trange, tqdm
import numpy as np
import copy
import xarray as xr
from ensembles.wasserstein import gaussian_w2_distance_distrax
import distrax as dx
import warnings
import jax
from jax import lax
import properscoring as ps

class AbstractWeight:
    """ An abstract class for weighting methods. These need to be instantiated (e.g., weight_method = AbstractWeight()) and implement by calling the weight (e.g, weghts = weight_method(args)).
    """    
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:
        raise NotImplementedError

    def __call__(
        self,
        process_models: ModelCollection,
        observations: ProcessModel = None,
        **kwargs
    ) -> xr.DataArray:
        """ Computes the weights for the given process models and observations.

        Args:
            process_models (ModelCollection): The models to be weighted. These models must have been emulated/fit e.g., using the GPDTW1D class.
            observations (ProcessModel, optional): The observations from which to weight the models. Defaults to None.

        Returns:
            weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
        """

        # Check that the observations are the same as the models
        if observations is not None:
            assert np.all(process_models.time == observations.time), "Time coordinates do not match between models and observations"
            assert len(process_models.time) == len(observations.time), "Time coordinates do not match between models and observations"
        # Check process models have been emulated
        for process_model in process_models.models:
            assert hasattr(process_model.distribution, '_dist'), "Distribution not defined - fit models first"
        
        return self._compute(
            process_models=process_models, observations=observations, **kwargs
        )


class LogLikelihoodWeight(AbstractWeight):
    """ A class for computing weights based on the log-likelihood of the observations given the model.

    Inherits:
        AbstractWeight
    """    
    def __init__(self, name: str = "LogLikelihoodWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self,
        process_models: ModelCollection,
        observations: ProcessModel,
        return_lls=False,
        standardisation_scheme=jnp.exp,
        standardisation_constant=1.,
    ) -> xr.DataArray:
        """ Computes weights based on the log-likelihood of the observations given the model.

        Args:
            process_models (ModelCollection): The models to be weighted. These models must have been emulated/fit e.g., using the GPDTW1D class.
            observations (ProcessModel): The observations from which to weight the models.
            return_lls (bool, optional): If True, returns the raw log-likelihoods. Defaults to False.
            standardisation_scheme (str, optional): A functions (np or jnp) with which to transform the weights after finding the log-likelihood. The helps avoid scaling issues and enforces positivity. Defaults to jnp.exp.
            standardisation_constant (float, optional): Scaling factor which modulates the strength of the log-likelihood weighting. Defaults to 1..

        Returns:
            weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
        """    

        model_lls = []
        for model in tqdm(process_models):
            distribution = model.distribution._dist
            log_likelihood = lambda x: distribution.log_prob(x)

            lls = []
            for obs_real in observations:
                # This is need because dx.Normal and dx.MultiVariate treat inputs differently
                if model.distribution.dist_type == dx.Normal:
                    ll_val = log_likelihood(obs_real.values.ravel())
                else:
                    ll_val = log_likelihood(
                        jnp.expand_dims(obs_real.values.ravel(), -1)
                    )
                lls.append(ll_val)
            # Collect the log-likelihoods across observations realisations (or products)
            lls_array = jnp.asarray(lls)
            lls_mean = jnp.mean(lls_array, axis=0)

            # Standardise the log-likelihoods - this helps avoid scaling issues and enforces positivity
            lls_mean = standardisation_scheme(standardisation_constant * lls_mean)

            # Place weights for a single model into a DataArray and collect these
            lls_mean_xarray = copy.deepcopy(
                model.model_data.isel(realisation=0)
            ).drop_vars("realisation")
            lls_mean_xarray.data = lls_mean.reshape(lls_mean_xarray.shape)
            lls_mean_xarray = lls_mean_xarray.assign_coords(model=model.model_name)
            model_lls.append(lls_mean_xarray)

        # Concatenate all model weights into a single DataArray
        model_lls = xr.concat(model_lls, dim="model")  # (n_reals, time)
        model_lls = model_lls.rename("Log-likelihoods")

        # Normalise the weights so they sum to 1 across the model dimension
        model_lls_sum = model_lls.sum("model")
        weights = model_lls / model_lls_sum

        weights = weights.rename("Log-likelihood weights")
        assert weights.shape == (len(process_models),) + obs_real.shape

        if return_lls:
            return weights, model_lls
        else:
            return weights


class InverseSquareWeight(AbstractWeight):
    """ A class for computing weights based on the the mean squared error between the models and observations.

    Inherits:
        AbstractWeight
    """ 
    def __init__(self, name: str = "InverseSquareWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:
        """Computes weights based on the inverse square difference between observations and models.

            Args:
                process_models (ModelCollection): The models to be weighted. These models must have been emulated/fit e.g., using the GPDTW1D class.
                observations (ProcessModel): The observations from which to weight the models.

            Returns:
                weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
            """

        weights = []
        for model in process_models:
            model_mean = model.mean_across_realisations
            obs_mean = observations.mean_across_realisations
            model_weight = (model_mean - obs_mean) ** -2
            model_weight = model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)

        weights = xr.concat(weights, dim="model")
        weights = weights.rename("Inverse square weights")

        # Normalise the weights so they sum to 1 across the model dimension
        weights = weights / weights.sum("model")
        assert (
            weights.time.size == model.time.size
        ), "Weight is not the same size as model. Check observations and model time coordinates match!"

        return weights


class UniformWeight(AbstractWeight):
    """ A class for producing uniform weights.

    Inherits:
        AbstractWeight
    """ 
    def __init__(self, name: str = "UniformWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection, observations: ProcessModel
    ) -> xr.DataArray:
        """Computes uniform weights for models i.e. 1/n_models.

            Args:
                process_models (ModelCollection): The models.

            Returns:
                weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
            """

        weights = []
        for model in process_models:
            model_weight = model.mean_across_realisations * 0 + 1.0 / len(
                process_models
            )
            model_weight = model_weight.assign_coords(model=model.model_name)
            weights.append(model_weight)

        weights = xr.concat(weights, dim="model")
        weights = weights.rename("Uniform weights")

        assert weights.time.size == model.time.size

        return weights

class ModelSimilarityWeight(AbstractWeight):
    """ A class for computing weights based on the similarity to other models. Highly similar models recieve a lower weight and vica versa.

    Inherits:
        AbstractWeight
    """
    def __init__(self, name: str = "ModelSimilarityWeight") -> None:
        super().__init__(name)

    @abc.abstractmethod
    def _compute(
        self, process_models: ModelCollection,
        mode: str = "single",
        observations: ProcessModel = None,
    ) -> xr.DataArray:
        """Computes weights based on inter-model similarity using the Wasserstein distance between model distributions.

            Args:
                process_models (ModelCollection): The models to be weighted. These models must have been emulated/fit e.g., using the GPDTW1D class.
                observations (ProcessModel): The observations from which to weight the models.
                mode (str): The mode in which to compute the weights. Options are: 'single' (default), 'temporal', or 'spatial'. In 'single' mode, the a single weight is produced for each model. In 'spatial' model, a weight is produced for each model at each spatial location. In temporal mode, a weight is produced for each model at each time step. There are computational advantages to using 'single' mode.

            Returns:
                weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
            """

        if mode == "single":
            if process_models[0].model_data.ndim > 2:
                warnings.warn('Mode "single" only really designed for small amounts of data. Kernel may crash. Try mode="spatial"')

            # Calculate the wasserstein distance between the models
            w2_dists = np.zeros(shape=(process_models.number_of_models, process_models.number_of_models)) * np.nan
            for i in trange(process_models.number_of_models):
                for j in range(process_models.number_of_models):
                    if isinstance(process_models[i].distribution._dist, dx.Normal):
                        full_cov = False
                    else:
                        full_cov = True
                    w2_model = gaussian_w2_distance_distrax(
                        process_models[i].distribution._dist,
                        process_models[j].distribution._dist,
                        full_cov=full_cov)
                    w2_dists[i, j] = w2_model

            # Collapse the wasserstein distance matrix into a vector
            w2_dists_vector = np.nanmean(w2_dists, axis=1)

            # Put weights into an xarray DataArray for continuity and dimension description
            weights_array = xr.DataArray(np.expand_dims(w2_dists_vector, -1), dims=['model', 'time'])
            weights_array = weights_array.rename("Model similarity weights")
            weights_array = weights_array.assign_coords(model=process_models.model_names)
            weights_array = weights_array.assign_coords(time=np.asarray([0]))


        elif mode == 'spatial':
            warnings.warn('Spatial method is experimental. Use with caution.')
            n_models = process_models.number_of_models
            n_lat = process_models[0].model_data.latitude.size
            n_lon = process_models[0].model_data.longitude.size
            w2_dists = np.zeros(shape=(n_models, n_models, n_lat, n_lon)) * np.nan

            for i in trange(n_models):
                for j in range(n_models):
                    for lat in range(n_lat):
                        for lon in range(n_lon):
                            if isinstance(process_models[i].distribution._dist, dx.Normal):
                                full_cov = False
                                dist = dx.Normal
                            else:
                                full_cov = True
                                dist = dx.MultiVariate
                            dist1 = dist(
                                process_models[i].distribution.mean[:, lat, lon], 
                                process_models[i].distribution.variance[:, lat, lon])
                            dist2 = dist(
                                process_models[j].distribution.mean[:, lat, lon], 
                                process_models[j].distribution.variance[:, lat, lon])
                            w2_model = gaussian_w2_distance_distrax(dist1, dist2, full_cov=full_cov)
                            w2_dists[i, j, lat, lon] = w2_model
            # Collapse the wasserstein distance matrix into a 3D matrix
            w2_dists_matrix = np.nanmean(w2_dists, axis=1)
            # Put weights into an xarray DataArray for continuity and dimension description
            weights_array = xr.DataArray(w2_dists_matrix, dims=['model', 'latitude', 'longitude']) 
            weights_array = weights_array.rename("Model similarity weights")
            weights_array = weights_array.assign_coords(model=process_models.model_names)
            weights_array = weights_array.assign_coords(longitude=process_models[0].model_data.longitude)
            weights_array = weights_array.assign_coords(latitude=process_models[0].model_data.latitude)

        elif mode == 'temporal':
            n_models = process_models.number_of_models
            n_times = process_models[0].model_data.time.size
            w2_dists = np.zeros(shape=(n_models, n_models, n_times)) * np.nan

            for i in trange(n_models):
                for j in range(n_models):
                    for t in range(n_times):
                        dist = dx.Normal
                        dist1 = dist(
                            process_models[i].distribution.mean[t].values.reshape(-1),
                            process_models[i].distribution.variance[t].values.reshape(-1))
                        dist2 = dist(
                            process_models[j].distribution.mean[t].values.reshape(-1),
                            process_models[j].distribution.variance[t].values.reshape(-1))
                        w2_model = gaussian_w2_distance_distrax(dist1, dist2, full_cov=False)
                        w2_dists[i, j, t] = w2_model
            # Collapse the wasserstein distance matrix into a 2D matrix
            w2_dists_matrix = np.nanmean(w2_dists, axis=1)
            # Put weights into an xarray DataArray for continuity and dimension description
            weights_array = xr.DataArray(w2_dists_matrix, dims=['model', 'time'])
            weights_array = weights_array.rename("Model similarity weights")
            weights_array = weights_array.assign_coords(model=process_models.model_names)
            weights_array = weights_array.assign_coords(time=process_models[0].model_data.time)

        else:
            raise ValueError('Mode must be "single", "spatial", or "temporal"')

        # Standardise the weights such that they sum to 1.
        weights_array = weights_array / weights_array.sum("model")
        
        return weights_array


class KSDWeight(AbstractWeight):
    """ A class for computing weights based on a Kernel Stein Discrepancy score.

    Inherits:
        AbstractWeight
    """
    def __init__(self, name: str = 'KernelSteinDiscrepancyWeight') -> None:
        super().__init__(name)

    def _compute(
        self,
        process_models: ModelCollection,
        observations: ProcessModel
        ) -> xr.DataArray:
        """Computes weights based on the Kernel Stein Discrepancy score.

            Args:
                process_models (ModelCollection): The models to be weighted. These models must have been emulated/fit e.g., using the GPDTW1D class.
                observations (ProcessModel): The observations from which to weight the models.

            Returns:
                weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
            """
    
        def k_0_fun(
            parm1: jnp.ndarray,
            parm2: jnp.ndarray,
            gradlogp1: jnp.ndarray,
            gradlogp2: jnp.ndarray,
            c: float = 1.0,
            beta: float = -0.5,
        ) -> jnp.ndarray:
            diff = parm1 - parm2
            dim = parm1.shape[0]
            imq_kernel_term = c**2 + jnp.dot(diff, diff)
            term1 = jnp.dot(gradlogp1, gradlogp2) * imq_kernel_term**beta
            term2 = -2 * beta * jnp.dot(gradlogp1, diff) * imq_kernel_term ** (beta - 1)
            term3 = 2 * beta * jnp.dot(gradlogp2, diff) * imq_kernel_term ** (beta - 1)
            term4 = -2 * dim * beta * (imq_kernel_term ** (beta - 1))
            term5 = -4 * beta * (beta - 1) * imq_kernel_term ** (beta - 2) * jnp.sum(jnp.square(diff))
            return term1 + term2 + term3 + term4 + term5

        _batch_k_0_fun_rows = jax.vmap(k_0_fun, in_axes=(None, 0, None, 0, None, None))

        @jax.jit
        def imq_KSD(samples: jnp.ndarray, grads: jnp.ndarray, c: float = 1., beta: float= -0.5) -> jnp.ndarray:
            # Inverse multiquadratic kernel
            N = samples.shape[0]

            def body_ksd(ksd_summand, x):
                my_sample, my_grad = x
                ksd_summand += jnp.sum(
                    _batch_k_0_fun_rows(my_sample, samples, my_grad, grads, c, beta)
                )
                return ksd_summand, None

            ksd_summand, _ = lax.scan(body_ksd, 0.0, (samples, grads))
            return jnp.sqrt(ksd_summand) / N

        assert len(process_models.time) == len(observations.time), "Time coordinates do not match between models and observations"
        assert hasattr(process_models[0].distribution, '_dist'), "Distribution not defined - fit models first"

        # Flatten observational data to feed into KSD
        observations_flat = observations.model_data.values.reshape(
            observations.n_realisations,
            observations.model_data.size // observations.n_realisations)

        # Want to save the KSD value per model (but flattened over other dims)
        output_shape = [process_models.number_of_models, observations.model_data.size // len(observations.model_data.realisation)]
        ksd_values = np.zeros(output_shape) * np.nan

        # Loop over models and compute KSD
        models_ksds = []
        for model_index in trange(process_models.number_of_models):
            ksd_values = []
            model = process_models[model_index]
            # Flatten the data
            model_mean = model.distribution._dist.mean()
            model_var = model.distribution._dist.variance()

            for i in range(len(model_mean)):
                target_density = dx.Normal(model_mean[i], model_var[i])
                obs_samples = observations_flat[:, i].reshape(-1, 1) # Needs to be (n, 1) shape
                grad_log_pi = jax.vmap(jax.grad(lambda x: target_density.log_prob(x).squeeze()))(obs_samples)
                ksd = imq_KSD(obs_samples, grad_log_pi)
                ksd_values.append(ksd)

            ksd_xarray = copy.deepcopy(
                model.model_data.isel(realisation=0)
            ).drop_vars("realisation")

            ksd_xarray.data = jnp.asarray(ksd_values).reshape(ksd_xarray.shape)
            ksd_xarray = ksd_xarray.assign_coords(model=model.model_name)
            models_ksds.append(ksd_xarray)

        # Put weights into an xarray DataArray for continuity and dimension description
        ksd_xarray = xr.concat(models_ksds, dim="model")
        ksd_xarray = ksd_xarray.rename("Kernel Stein Discrepancy")
        ksd_inverse = 1 / ksd_xarray 

        # Standardise the weights such that they sum to 1.
        ksd_sum = ksd_inverse.sum(dim="model")
        ksd_weights = ksd_inverse / ksd_sum
        ksd_weights = ksd_weights.rename("Kernel Stein Discrepancy weights")

        return ksd_weights


class CRPSWeight(AbstractWeight):
    """ A class for computing weights based on the Continuous ranked probability score.

    Inherits:
        AbstractWeight
    """
    def __init__(self, name: str = 'ContinuousRankedProbabilityScoreWeight') -> None:
        super().__init__(name)

    def _compute(
        self,
        process_models: ModelCollection,
        observations: ProcessModel
        ) -> xr.DataArray:
        """Computes weights based on the Continuous ranked probability score.

            Args:
                process_models (ModelCollection): The models to be weighted. These models must have been emulated/fit e.g., using the GPDTW1D class.
                observations (ProcessModel): The observations from which to weight the models.

            Returns:
                weights (xr.DataArray): The calculated weights, within a xr.DataArray to maintain dimensions.
            """


        def crps_score(samples, posterior: dx.Distribution) -> jnp.ndarray:
            mu, sigma = posterior.mean(), posterior.stddev()
            return np.mean([ps.crps_gaussian(obs, mu=mu, sig=sigma) for obs in samples])

        assert len(process_models.time) == len(observations.time), "Time coordinates do not match between models and observations"
        assert hasattr(process_models[0].distribution, '_dist'), "Distribution not defined - fit models first"

        # Flatten observational data
        observations_flat = observations.model_data.values.reshape(
            observations.n_realisations,
            observations.model_data.size // observations.n_realisations)

        # Loop over models and compute CRPS
        models_crpss = []
        for model_index in trange(process_models.number_of_models):
            crps_values = []
            model = process_models[model_index]
            # Flatten the data
            model_mean = model.distribution._dist.mean()
            model_var = model.distribution._dist.variance()

            for i in range(len(model_mean)):
                target_density = dx.Normal(model_mean[i], model_var[i])
                obs_samples = observations_flat[:, i].reshape(-1, 1) # Needs to be (n, 1) shape
                crps = crps_score(obs_samples, target_density)

                crps_values.append(crps)

            crps_xarray = copy.deepcopy(
                model.model_data.isel(realisation=0)
            ).drop_vars("realisation")

            crps_xarray.data = np.asarray(crps_values).reshape(crps_xarray.shape)
            crps_xarray = crps_xarray.assign_coords(model=model.model_name)
            models_crpss.append(crps_xarray)

        # Put weights into an xarray DataArray for continuity and dimension description
        crps_xarray = xr.concat(models_crpss, dim="model")
        crps_xarray = crps_xarray.rename("Continuous Ranked Probability Score")
        crps_inverse = 1 / crps_xarray 

        # Standardise the weights such that they sum to 1.
        crps_sum = crps_inverse.sum(dim="model")
        crps_weights = crps_inverse / crps_sum
        crps_weights = crps_weights.rename("Continuous Ranked Probability Scores weights")

        return crps_weights







