from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.context import FitContext
from yadlt.log import setup_logger
from yadlt.plotting.plot_covariance import (
    plot_cov_compare_tr_an,
    plot_covariance_decomposition,
    plot_diag_error_compare_an_tr,
    plot_diag_error_decomposition,
)
from yadlt.plotting.plot_distance import (
    plot_distance_from_input,
    plot_distance_from_train,
)
from yadlt.plotting.plot_evolution_pdf import (
    plot_evolution_from_initialisation,
    plot_evolution_vs_trained,
    plot_Q_directions,
)
from yadlt.plotting.plot_u_v_contribution import plot_u_v_contributions

logger = setup_logger()
logger.setLevel(logging.INFO)

SEED = 1423413

REF_EPOCH = 20000
FITNAMES = [
    "250713-01-L0-nnpdf-like",
    "250713-02-L1-nnpdf-like",
    "250713-03-L2-nnpdf-like",
]
CONFIGS = [
    {
        "scale": "linear",
        "divide_by_x": False,
        "ylim": None,
        "xlim": None,
        "prefix": "xT3",
    },  # Linear xT3
    {
        "scale": "linear",
        "divide_by_x": True,
        "ylim": (-1, 3),
        "xlim": None,
        "prefix": "T3",
    },  # Linear T3
    {
        "scale": "log",
        "divide_by_x": False,
        "ylim": None,
        "xlim": (1e-3, 1),
        "prefix": "xT3",
    },  # Log xT3
    {
        "scale": "log",
        "divide_by_x": True,
        "ylim": (-10, 10),
        "xlim": (1e-3, 1),
        "prefix": "T3",
    },  # Log T3
]

COLOR_TS = "C0"
COLOR_AS = "C1"
COLOR_AS_REF1 = "C2"
COLOR_AS_REF2 = "C3"
COLOR_AS_REF3 = "C4"


def main():
    parser = ArgumentParser(description="Compute and plot PDFs.")
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force serialization of the data even if it already exists.",
    )
    args = parser.parse_args()
    DIR = Path(args.plot_dir) if args.plot_dir else Path(__file__).parent / "plots"
    PLOT_DIR = DIR / "analytical_solution"

    for fitname in FITNAMES:
        context = FitContext(fitname, force_serialize=parser.parse_args().force)
        datatype = context.get_config("metadata", "arguments")["data"]

        for config in CONFIGS:
            scale = config["scale"]
            divide_by_x = config["divide_by_x"]
            xlim = config["xlim"]
            ylim = config["ylim"]
            prefix = config["prefix"]

            ######################################################
            #  Evolution from initialisation and different epochs
            #######################################################
            logger.info(
                f"Plotting evolution from initialisation 1 for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            plot_evolution_from_initialisation(
                context,
                ref_epoch=REF_EPOCH,
                epochs=[700, 5000, 20000],
                colors=[COLOR_TS, COLOR_AS_REF1, COLOR_AS_REF2, COLOR_AS_REF3],
                ax_specs=[
                    {"set_xscale": scale, "set_xlim": xlim, "set_ylim": ylim},
                    {},
                ],
                divide_by_x=divide_by_x,
                plot_dir=PLOT_DIR / f"{prefix}/evolution/from_f0/{datatype}/{scale}",
                filename=f"evolution_epochs_700_5000_20000_{datatype}_{scale}.pdf",
                save_fig=True,
                ratio_label=r"$\textrm{Ratio to}$" + "\n" + r"$\rm{Trained}$",
            )

            ######################################################
            #  Evolution from initialisation and last epoch
            #######################################################
            logger.info(
                f"Plotting evolution from initialisation 2 for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            plot_evolution_from_initialisation(
                context,
                ref_epoch=REF_EPOCH,
                epochs=[50000],
                show_true=True,
                ax_specs=[
                    {"set_xscale": scale, "set_xlim": xlim, "set_ylim": ylim},
                    {},
                ],
                divide_by_x=divide_by_x,
                colors=[COLOR_TS, COLOR_AS],
                plot_dir=PLOT_DIR / f"{prefix}/evolution/from_f0/{datatype}/{scale}",
                filename=f"evolution_epoch_50000_{datatype}_{scale}.pdf",
                save_fig=True,
                ratio_label=r"$\textrm{Ratio to}$" + "\n" + r"$\rm{True}$",
            )

            ################################################
            #  U and V contributions
            ################################################
            logger.info(
                f"Plotting U and V contributions for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            epochs_to_plot = [0, 10, 50, 100, 20000]
            for epoch in epochs_to_plot:
                plot_u_v_contributions(
                    context,
                    ref_epoch=REF_EPOCH,
                    ev_epoch=epoch,
                    seed=SEED,
                    ax_specs=[
                        {"set_xscale": scale, "set_xlim": xlim, "set_ylim": ylim},
                        {},
                    ],
                    divide_by_x=divide_by_x,
                    save_fig=True,
                    plot_dir=PLOT_DIR
                    / f"{prefix}/u_v_decomposition/{datatype}/{scale}",
                    filename=f"evolution_u_v_{epoch}_{datatype}_{scale}.pdf",
                    ratio_label=r"$\textrm{Ratio to}$" + "\n" + r"$\rm{Trained}$",
                )

            ################################################
            #  Evolution Analytical Vs Trained
            ################################################
            logger.info(
                f"Plotting evolution vs trained for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            epochs_to_plot = [0, 500, 1000, 10000, 20000]
            for epoch in epochs_to_plot:
                # Evolution plot
                plot_evolution_vs_trained(
                    context,
                    ref_epoch=REF_EPOCH,
                    legend_title=rf"$\textrm{{{datatype} data}}$",
                    colors=[COLOR_TS, COLOR_AS],
                    epoch=epoch,
                    seed=SEED,
                    ax_specs=[
                        {"set_xscale": scale, "set_xlim": xlim, "set_ylim": ylim},
                        {},
                    ],
                    divide_by_x=divide_by_x,
                    show_true=False,
                    save_fig=True,
                    plot_dir=PLOT_DIR
                    / f"{prefix}/evolution/tr_vs_an/{datatype}/{scale}",
                    filename=f"evolution_vs_trained_epoch_{epoch}_{datatype}_{scale}.pdf",
                    ratio_label=r"$\textrm{Ratio to}$" + "\n" + r"$\rm{Trained}$",
                )

            ################################################
            #  Distance from input and from training
            ################################################
            logger.info(
                f"Plotting distance from input for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            epochs_to_plot = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            for epoch in epochs_to_plot:
                plot_distance_from_input(
                    context,
                    ref_epoch=20000,
                    epoch=epoch,
                    seed=SEED,
                    show_std=True,
                    ax_specs={"set_xscale": scale, "set_xlim": xlim},
                    divide_by_x=divide_by_x,
                    save_fig=True,
                    filename=f"distance_from_input_epoch_{epoch}_{datatype}_{scale}.pdf",
                    plot_dir=PLOT_DIR
                    / f"{prefix}/distance_from_input/{datatype}/{scale}",
                    title=rf"$\textrm{{{datatype} data}}~T={epoch}$",
                )

                plot_distance_from_train(
                    context,
                    ref_epoch=20000,
                    epoch=epoch,
                    seed=SEED,
                    show_std=True,
                    save_fig=True,
                    ax_specs={"set_xscale": scale, "set_xlim": xlim},
                    divide_by_x=divide_by_x,
                    filename=f"distance_plot_from_training_epoch_{epoch}_{datatype}_{scale}.pdf",
                    plot_dir=PLOT_DIR
                    / f"{prefix}/distance_from_training/{datatype}/{scale}",
                    title=rf"$\textrm{{{datatype} data}}~T={epoch}$",
                )

            ################################################
            #  Plot the covariance
            ################################################
            # plot_covariance_decomposition(
            #     context,
            #     ref_epoch=20000,
            #     epochs=[0, 1, 100],
            #     seed=SEED,
            #     save_fig=True,
            #     plot_dir=PLOT_DIR / f"covariance/matrix/decomposition/{datatype}",
            # )

            # plot_cov_compare_tr_an(
            #     context,
            #     ref_epoch=20000,
            #     epochs=[0, 500, 1000, 10000],
            #     seed=SEED,
            #     save_fig=True,
            #     plot_dir=PLOT_DIR / f"covariance/matrix/analytical_trained_comparison/{datatype}",
            # )

            ################################################
            #  Plot the diagonal error decomposition
            ################################################
            logger.info(
                f"Plotting diagonal error decomposition for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            epochs = [0, 50, 100, 500, 1000, 10000, 20000]
            common_spec = {
                "alpha": 0.5,
                "marker": "X",
                "markersize": 5,
                "lw": 2.0,
                "capthick": 2,
                "capsize": 3,
                "linestyle": "None",
            }
            for epoch in epochs:
                plot_diag_error_decomposition(
                    context,
                    ref_epoch=20000,
                    epoch=epoch,
                    common_plt_spec=common_spec,
                    seed=SEED,
                    ax_specs={"set_xscale": scale, "set_ylim": ylim},
                    divide_by_x=divide_by_x,
                    title=rf"$\textrm{{Decomposition at }} T = {epoch}$",
                    xlabel=r"$x$",
                    ylabel=r"$\sigma~(\sqrt{C_{ii}})$",
                    save_fig=True,
                    plot_dir=PLOT_DIR
                    / f"{prefix}/covariance/diagonal/decomposition/{datatype}/{scale}",
                    filename=f"diag_error_decomposition_epoch_{epoch}_{datatype}_{scale}.pdf",
                )

            ################################################
            #  Plot the diagonal error training vs analytical
            ################################################
            logger.info(
                f"Plotting diagonal error training vs analytical for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            for epoch in [0, 500, 1000, 2000, 20000]:
                plot_diag_error_compare_an_tr(
                    context,
                    ref_epoch=20000,
                    epoch=epoch,
                    common_plt_spec=common_spec,
                    seed=SEED,
                    ax_specs={"set_xscale": scale, "set_ylim": ylim},
                    divide_by_x=divide_by_x,
                    title=rf"$\textrm{{Standard deviation at }} T = {epoch}$",
                    xlabel=r"$x$",
                    ylabel=r"$\sigma~(\sqrt{C_{ii}})$",
                    save_fig=True,
                    plot_dir=PLOT_DIR
                    / f"{prefix}/covariance/diagonal/analytical_trained_comparison/{datatype}/{scale}",
                    filename=f"diag_error_an_tr_compar_epoch_{epoch}_{datatype}_{scale}.pdf",
                )

            ################################################
            #  Plot the vectors of the map Q
            ################################################
            logger.info(
                f"Plotting Q directions for {fitname} with {datatype} data, scale={scale}, divide_by_x={divide_by_x}"
            )
            ylim_q = (-10, 10) if ylim is not None else None
            for epoch in [0, 500, 1000, 5000, 10000, 20000, 50000]:
                plot_Q_directions(
                    context,
                    ref_epoch=epoch,
                    ranks=[1, 2, 3, 4],
                    colors=["C0", "C1", "C2", "C3"],
                    ax_specs={
                        "set_ylabel": r"$\pmb{q}(x)$",
                        "set_xscale": scale,
                        "set_xlim": xlim,
                        "set_ylim": ylim_q,
                    },
                    divide_by_x=divide_by_x,
                    title=rf"$T = {epoch}$",
                    save_fig=True,
                    filename=f"q_components_epoch_{epoch}_{datatype}_{scale}.pdf",
                    plot_dir=PLOT_DIR / f"{prefix}/q_components/{datatype}/{scale}",
                )

        context.remove_self()


if __name__ == "__main__":
    main()
