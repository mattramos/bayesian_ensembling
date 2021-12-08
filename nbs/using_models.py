# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''bayesian_ensembles'': conda)'
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# ## Using Models
#
# In the Bayesian Emulators package a series of models are supplied. In this notebook, we show how these models can be used in isolation. 

import ensembles as es
import numpy as np
import gpflow
import matplotlib.pyplot as plt

# ## Data
#
# We'll first simulate some data from a latent function that we'll later try to recover. We'll perturb this function with some iid Gaussian noise and that willl give us our inputs $x$ and corresponding response variable $y$.

# +
rng = np.random.RandomState(123)
x = np.linspace(-3., 3., num = 50).reshape(-1, 1)
x_test = np.linspace(-3.5, 3.5, num=10000).reshape(-1,1)
f = lambda x: np.sin(3*x) + np.sin(5*x)
y = f(x) + rng.normal(loc = 0., scale = 0.2, size=x.shape)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_test, f(x_test))
ax.plot(x, y, 'o')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# -

# ## Model
#
# Great, with a dataset defined we are now free to define our models and fit them. In this package, we supply a high-level interface that follows the SKLearn flavour of modelling with `.fit()` and `.predict()` method. We'll be deomonstrating the use of a GP regression model here. We can define it as follows

model = es.ConjugateGP(kernel = gpflow.kernels.RBF())

# ### Inference
#
# With a model instantiated, the next step is to optimise our model's hyperparameters. To do this, we require a dataset to be supplied along with a dictionary of parameter values. To make this process easier, we supply a series of default values that can be accessed by the following command.

params = es.config.GPRParameters().to_dict()
params

# We'll now use these parameters to fit our model according to the data we simulated above.

model.fit(X = x, y = y, params=params)

# ### Predictions
#
# We'll now condition on the observed dataset to query our model at a series of test locations. This is achieved through the `predict` method as follows

mu, sigma = model.predict(X=x_test, params=params)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y, 'o', color='tab:blue')
ax.plot(x_test, mu.numpy(), color='tab:orange')
ax.fill_between(x_test.ravel(), mu.numpy().ravel() - np.sqrt(sigma.numpy().ravel()), mu.numpy().ravel() + np.sqrt(sigma.numpy().ravel()), alpha=0.3, color='tab:orange')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(xlabel=r'x', ylabel=r'f(x)', title=f'{model.name} model predictions')

# ## System configuration

# %load_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
