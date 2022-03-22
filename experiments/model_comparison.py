import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gpflow

from ensembles.plotters import plot_individual_preds, plot_group_pred

tf.config.run_functions_eagerly(True)
import ensembles as es
from ensembles import HSGP

SEED = 30


if __name__ == "__main__":
    args = es.utils.build_parser()
    xlims = (-args["xlims"], args["xlims"])
    # Simulate data
    X, Y, Ys = es.simulate_data(
        n_obs=args["n_obs"],
        n_realisations=args["n_realisation"],
        true_kernel=args["true_kernel"],
        noise_lims=(0.1, args["noise"]),
        xlims=xlims,
        seed_value=SEED,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    es.plotters.plot_realisations(X, Y, Ys, ax, filename="sim_data.png")

    # Define HGP model
    kernel = gpflow.kernels.Matern32()
    Z = np.linspace(*xlims, args["n_inducing"]).reshape(-1, 1)
    hsgp = HSGP(
        (X, Ys),
        group_kernel=args["group_kernel"],
        individual_kernel=args["individual_kernel"],
        inducing_points=Z,
    )
    gpflow.utilities.set_trainable(hsgp.inducing_points, args["train_inducing"])

    # Fit
    params = es.SGPRParameters().to_dict()
    params["learning_rate"] = 0.1
    params["optim_nits"] = args["optim_nits"]
    tf.config.run_functions_eagerly(False)
    hsgp.fit(params)
    tf.config.run_functions_eagerly(True)

    # Evaluate
    Xte = np.linspace(xlims[0] * 1.1, xlims[1] * 1.1, args["n_obs"]).reshape(-1, 1)
    indi_preds = [hsgp.predict_individual(Xte, idx) for idx in range(args["n_realisation"])]
    plot_individual_preds(
        Xte=Xte, truth=Ys, individual_preds=indi_preds, filename="individual_preds.png"
    )

    group_mean, group_var = hsgp.predict_group(Xte)
    group_mean = group_mean.numpy().squeeze()
    group_std = np.sqrt(group_var.numpy().squeeze())
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_group_pred(
        mu=group_mean,
        sigma=group_std,
        Xte=Xte.squeeze(),
        Xtr=X.squeeze(),
        latent_y=Y,
        realisations=Ys,
        ax=ax,
        filename="group_preds.png",
    )
