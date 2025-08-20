"""Module to plot the distance between PDFs."""

from yadlt.context import FitContext
from yadlt.distribution import Distribution
from yadlt.plotting.plotting import produce_distance_plot
from yadlt.utils import (
    evaluate_from_initialisation,
    load_and_evaluate_model,
)


def plot_distance_from_input(
    context: FitContext,
    ref_epoch: int = 0,
    epoch: int = 0,
    epoch_tr: int = None,
    seed: int = 0,
    **plot_kwargs,
):
    """Produce a distance plot wrt the trained solution."""
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    # last_epoch = context.get_config("replicas", "common_epochs")[-1]
    fk_grid = context.load_fk_grid()
    f_input = context.load_f_bcdms()

    # Wrap the input PDF around a Distribution object
    f_input_dist = Distribution(
        name=r"$\textrm{Input PDF}$", size=1, shape=f_input.shape
    )
    f_input_dist.add(f_input)

    # Load trained solution at the end of training
    if epoch_tr is None:
        epoch_tr = epoch
    xT3_training = load_and_evaluate_model(context, epoch_tr)
    xT3_training.set_name(rf"$\textrm{{TS @ }} T = {{{epoch_tr}}}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_t = evaluate_from_initialisation(context, ref_epoch=ref_epoch, seed=seed)
    xT3_at_epoch = xT3_t(epoch * learning_rate)
    xT3_at_epoch.set_name(rf"$\textrm{{AS @ }} T = {{{epoch}}}$")

    produce_distance_plot(
        fk_grid,
        [f_input_dist, xT3_at_epoch, xT3_training],
        normalize_to=1,
        ylabel=r"$\textrm{Distance from input}$",
        figsize=(10, 8),
        **plot_kwargs,
    )


def plot_distance_from_train(
    context: FitContext,
    ref_epoch: int = 0,
    epoch: int = 0,
    seed: int = 0,
    **plot_kwargs,
):
    """Produce a distance plot wrt the trained solution."""
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    # last_epoch = context.get_config("replicas", "common_epochs")[-1]
    fk_grid = context.load_fk_grid()
    f_input = context.load_f_bcdms()

    # Wrap the input PDF around a Distribution object
    f_input_dist = Distribution(
        name=r"$\textrm{Input PDF}$", size=1, shape=f_input.shape
    )
    f_input_dist.add(f_input)

    # Load trained solution at the end of training
    xT3_training = load_and_evaluate_model(context, epoch)
    xT3_training.set_name(rf"$\textrm{{TS @ }} T = {{{epoch}}}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_t = evaluate_from_initialisation(context, ref_epoch=ref_epoch, seed=seed)
    xT3_at_epoch = xT3_t(epoch * learning_rate)
    xT3_at_epoch.set_name(rf"$\textrm{{AS @ }} T = {{{epoch}}}$")

    produce_distance_plot(
        fk_grid,
        [xT3_training, xT3_at_epoch],
        normalize_to=1,
        ylabel=r"$\textrm{Distance from training}$",
        figsize=(10, 8),
        **plot_kwargs,
    )
