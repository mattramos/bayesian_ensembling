Installation
======================

Stable version
-----------------

To install the latest stable release using :code:`pip` execute the following:

.. code-block:: bash

    pip install bayesian_ensembles

GPU support
^^^^^^^^^^^^^^^^^^^

GPU support is enabled through proper configuration of the underlying `Jax <https://github.com/google/jax>`_ installation. CPU enabled forms of both packages are installed as part of the GPJax installation. For GPU Jax support, the following command should be run

.. code-block:: bash

    # Specify your installed CUDA version.
    CUDA_VERSION=11.0
    pip install jaxlib

Then, within a Python shell run

.. code-block:: python

    import jaxlib
    print(jaxlib.__version__)

