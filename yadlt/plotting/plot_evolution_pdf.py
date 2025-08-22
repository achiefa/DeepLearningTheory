"""Module to plot the complete evolution of PDFs."""

from fileinput import filename

from yadlt.context import FitContext
from yadlt.plotting.plotting import produce_pdf_plot, produce_plot
from yadlt.utils import (
    evaluate_from_initialisation,
    evaluate_from_ref_function,
    load_and_evaluate_model,
    load_data,
)


def plot_evolution_from_initialisation(
    context: FitContext,
    ref_epoch: 0,
    epochs: int = 0,
    seed: int = 0,
    show_true: bool = False,
    **plot_kwargs,
):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    common_epochs = context.get_config("replicas", "common_epochs")
    f_bcdms = context.load_f_bcdms()
    fk_grid = context.load_fk_grid()

    # Load trained solution
    xT3_training = load_and_evaluate_model(context, -1)
    xT3_training.set_name(rf"$\textrm{{TS @ }} T={common_epochs[-1]}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_t = evaluate_from_initialisation(context, ref_epoch=ref_epoch, seed=seed)

    # Evolve the solution for each epoc
    grids_list = []
    for epoch in epochs:
        if epoch == -1:
            epoch = common_epochs[-1]

        evolution_time = epoch * learning_rate
        tmp = xT3_t(evolution_time)
        tmp.set_name(rf"$\textrm{{AS @ }} T =  {{{epoch}}}$")
        grids_list.append(tmp)

    normalize_to = 1
    if show_true:
        add_grid_dict = {
            "mean": f_bcdms,
            "spec": {
                "linestyle": "--",
                "label": r"$\textrm{True function}$",
                "color": "black",
            },
        }
        normalize_to = -1

    ax_specs_ratio = {"set_ylim": (0.8, 1.2)}
    if plot_kwargs.get("ax_specs", None) is not None:
        plot_kwargs["ax_specs"][1] = plot_kwargs["ax_specs"][1] | ax_specs_ratio
    else:
        plot_kwargs["ax_specs"] = [None, ax_specs_ratio]

    # Check the colors, if give, are compatible with the number of grids
    if "colors" in plot_kwargs:
        colors = plot_kwargs["colors"]
        if len(colors) != len(grids_list) + 1:
            raise ValueError(
                "Please provide exactly one color for the training solution and one for each analytical solution."
            )

    produce_pdf_plot(
        fk_grid,
        [xT3_training, *grids_list],
        normalize_to=normalize_to,
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}$",
        additional_grids=[add_grid_dict] if show_true else None,
        xlabel=r"$x$",
        ylabel=r"$xT3(x)$",
        **plot_kwargs,
    )


def plot_evolution_from_ref(
    context: FitContext,
    ref_epoch: int = 0,
    tr_epoch: int = 0,
    show_ratio: bool = True,
    **plot_kwargs,
):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK.
    """
    metadata = context.get_config("metadata", "arguments")
    learning_rate = float(metadata["learning_rate"])

    # Compute evolution epochs and training time
    residual_epochs = ref_epoch + tr_epoch
    t = residual_epochs * learning_rate

    # Load trained solution (end of training)
    xT3_training = load_and_evaluate_model(context, residual_epochs)
    xT3_training.set_name(rf"$\textrm{{TS @ }}T = {{{residual_epochs}}}$")

    # Prepare evolution from reference function
    xT3_ref_t = evaluate_from_ref_function(context, ref_epoch=ref_epoch)
    xT3_t = xT3_ref_t(t)
    xT3_t.set_name(rf"$\textrm{{AS @ }}T = {{{residual_epochs}}}$")

    # Check the colors, if give, are compatible with the number of grids
    if "colors" in plot_kwargs:
        colors = plot_kwargs["colors"]
        if len(colors) != 2:
            raise ValueError("Please provide exactly two colors for the plots.")

    if show_ratio:
        produce_pdf_plot(
            context.load_fk_grid(),
            [xT3_training, xT3_t],
            normalize_to=1,
            title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm trained)}}_{{T_{{\rm ref}}}}$",
            **plot_kwargs,
        )
    else:
        produce_plot(
            context.load_fk_grid(),
            [xT3_training, xT3_t],
            title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm trained)}}_{{T_{{\rm ref}}}}$",
            **plot_kwargs,
        )


def plot_evolution_vs_trained(
    context: FitContext,
    ref_epoch: 0,
    epoch: int = 0,
    seed: int = 0,
    show_true: bool = False,
    **plot_kwargs,
):
    """Plot the PDF comparison from a random initialised
    model using the frozen NTK. At each epoch, the analytical solution
    is compared to the trained solution.

    Args:
        context (FitContext): The context containing the fit information.
        ref_epoch (int): The reference epoch for the evolution.
        epochs (list[int]): List of epochs to plot.
        seed (int): Random seed for initialisation.
        show_true (bool): Whether to show the true function in the plot.
        **plot_kwargs: Additional keyword arguments for the plot.
    """
    learning_rate = float(context.get_config("metadata", "arguments")["learning_rate"])
    common_epochs = context.get_config("replicas", "common_epochs")
    f_bcdms = context.load_f_bcdms()
    fk_grid = context.load_fk_grid()

    # Load trained solution
    xT3_training = load_and_evaluate_model(context, epoch)
    xT3_training.set_name(rf"$\textrm{{TS @ }} T =  {{{epoch}}}$")

    # Prepare evolution from initialisation and NTK frozen
    xT3_t = evaluate_from_initialisation(context, ref_epoch=ref_epoch, seed=seed)

    # Evolve the solution for each epoch
    if epoch == -1:
        epoch = common_epochs[-1]

    evolution_time = epoch * learning_rate
    grid = xT3_t(evolution_time)
    grid.set_name(rf"$\textrm{{AS @ }} T =  {{{epoch}}}$")

    if show_true:
        add_grid_dict = {
            "mean": f_bcdms,
            "spec": {
                "linestyle": "--",
                "label": r"$\textrm{True function}$",
                "color": "black",
            },
        }

    ax_specs_ratio = {"set_ylim": (0.8, 1.2)}
    if plot_kwargs.get("ax_specs", None) is not None:
        plot_kwargs["ax_specs"][1] = plot_kwargs["ax_specs"][1] | ax_specs_ratio
    else:
        plot_kwargs["ax_specs"] = [None, ax_specs_ratio]

    # Check the colors, if give, are compatible with the number of grids
    if "colors" in plot_kwargs:
        colors = plot_kwargs["colors"]
        if len(colors) != 2:
            raise ValueError("Please provide exactly two colors for the plots.")

    produce_pdf_plot(
        fk_grid,
        [xT3_training, grid],
        normalize_to=1,
        title=rf"$T_{{\rm ref}} = {{{ref_epoch}}}, \quad f_0 = f^{{(\rm init)}}$",
        additional_grids=[add_grid_dict] if show_true else None,
        xlabel=r"$x$",
        ylabel=r"$xT3(x)$",
        **plot_kwargs,
    )


def plot_Q_directions(
    context: FitContext,
    ref_epoch: 0,
    ranks: list[int],
    colors: list[str],
    **plot_kwargs,
):
    """Plot the vectors in Q, which are the combinations
    of the NTK directions and the FK table."""
    common_epochs = context.get_config("replicas", "common_epochs")
    epoch_idx = -1 if ref_epoch == -1 else common_epochs.index(ref_epoch)
    Q = context.Q_by_epoch[epoch_idx]
    fk_grid = context.load_fk_grid()

    grids = []

    for idx in ranks:
        # Slice the distribution and select the n-th vector
        grid = Q.slice((slice(None), idx - 1))
        grid.set_name(rf"$\pmb{{q}}^{{({idx})}}$")

        grids.append(grid)

    produce_plot(
        fk_grid,
        grids,
        xlabel=r"$x$",
        ylabel=r"$\pmb{q}$",
        # labels=fitlabels,
        colors=colors,
        **plot_kwargs,
    )
