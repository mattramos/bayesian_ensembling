# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''bayesian_ensembles'': conda)'
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from hgpmoe import HierarchicalGP, pred_x
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
import seaborn as sns

sns.set_style('whitegrid')

# %% [markdown]
# ## Define experimnental constants

# %%
SEED = 42
n = 100
n_clusters = 3
n_realisations = 12
noise_limits = (0.01, 0.5)
xlims = (0., 10.)
jitter_amount = 1e-6
rng = np.random.RandomState(SEED)
tfp_seed = tfp.random.sanitize_seed(SEED)


# %% [markdown]
# ## Define the _true_ latent process

# %%
X = np.linspace(*xlims, n).reshape(-1, 1)
true_kernel = gpflow.kernels.RBF()
Kxx = true_kernel(X) + tf.cast(tf.eye(n)*jitter_amount, dtype=tf.float64)
latent_y = tfp.distributions.MultivariateNormalTriL(np.zeros(n), tf.linalg.cholesky(Kxx)).sample(seed=tfp_seed)

# %% [markdown]
# ## Simulate noisy realisations of the process

# %%
noise_terms = np.random.uniform(*noise_limits, size=n_realisations)
realisations = []

for noise in noise_terms:
    sample_y = latent_y.numpy() + rng.normal(loc=0., scale=noise, size=latent_y.numpy().shape)
    realisations.append(sample_y)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
[ax.plot(X, r, alpha=0.3, color='tab:blue') for r in realisations]
ax.plot(X, latent_y, color='tab:orange')
plt.show()

# %% [markdown]
# ## Model defining
#
# We'll give each realisation a random cluster assignment

# %%
Xstack = [X.ravel()] * n_realisations
cluster_assignments = rng.randint(low = 0, high = n_realisations, size=n_realisations, dtype=np.int64).tolist()

# %%
hgp = HierarchicalGP((Xstack, realisations), cluster_assignments)

# %% [markdown]
# ## Model fitting
#
# We'll now optimise our model using the basic Scipy implementation of LBFGS

# %%
res = gpflow.optimizers.Scipy().minimize(hgp.training_loss, hgp.trainable_variables, options=dict(maxiter=1000))
assert res['success']

# %%
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(20, 13))
for i, ax in zip(range(6), axes.ravel()):
    pred_x(hgp, i, 
        Xstack, realisations,
        cluster_assignments, ax=ax, xlims=xlims, n_test_points=1)
plt.tight_layout()
plt.savefig('simulated_figs/individual_preds.pdf')
plt.savefig('simulated_figs/individual_preds.png')


# %% [markdown]
# ## Group-level prediction

# %%
def predict_latent(model:HierarchicalGP, Xnew, patient_idx, cluster_idx):
    Xi = model.X[patient_idx]
    Xi = tf.expand_dims(Xi, axis=-1)
    Yi = tf.expand_dims(model.Y[patient_idx], axis=-1)
    err = Yi - model.mean_function(Xi)
    model.K_group_list[cluster_idx](X)

    g = model.C_assignments[patient_idx]
    g_idx, = np.where(model.G == g)
    g_idx = g_idx[0]
    print("Xi: ", Xi.shape)
    print("Xnew: ", Xnew.shape)

    kmm = model.K_group_list[g_idx](Xi)
    knn = model.K_group_list[g_idx](Xnew, full_cov=False)
    kmn = model.K_group_list[g_idx](Xi, Xnew)
    kmm_plus_s = model._add_noise_cov(kmm)

    # print("kmm: ", kmm.shape)
    # print("knn: ", knn.shape)
    # print("kmn: ", kmn.shape)
    # print("kmm_plus_s: ", kmm_plus_s.shape)
    # print("err: ", err.shape)

    conditional = gpflow.conditionals.base_conditional
    f_mean_zero, f_var = conditional(
        kmn, kmm_plus_s, knn, err, full_cov=False, white=False
    ) 
    f_mean = f_mean_zero + model.mean_function(Xnew)
    return f_mean, f_var


# %%
patient_idx = 0
latent_mu, latent_sigma = predict_latent(hgp, tf.reshape(Xstack[0], (-1,1)), patient_idx, cluster_assignments[patient_idx])

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(X, latent_y, color='tab:blue', label='True latent fn.')
[ax.plot(X, r, alpha=0.1, color='tab:blue') for r in realisations]
ax.plot(Xstack[0].numpy(), latent_mu, color = 'tab:orange', label='Group latent fn.')
ax.fill_between(Xstack[0].numpy().ravel(), latent_mu.numpy().ravel() - np.sqrt(latent_sigma.numpy().ravel()), latent_mu.numpy().ravel() + np.sqrt(latent_sigma.numpy().ravel()), color = 'tab:orange', alpha=0.3, label=r'$1\sigma$')
ax.plot(X, [-3]*X.shape[0], '|', label='Observations', color='black')
ax.legend(loc='best')
plt.savefig('simulated_figs/group_preds.pdf')
plt.savefig('simulated_figs/group_preds.png')

# %%
