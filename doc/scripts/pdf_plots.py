from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.context import FitContext
from yadlt.log import setup_logger
from yadlt.plotting.plot_covariance import (
    plot_cov_compare_tr_an,
    plot_covariance_decomposition,
    plot_diag_error_decomposition,
)
from yadlt.plotting.plot_distance import (
    plot_distance_from_input,
    plot_distance_from_train,
)
from yadlt.plotting.plot_evolution_pdf import (
    plot_evolution_from_initialisation,
    plot_evolution_vs_trained,
)
from yadlt.plotting.plot_u_v_contribution import plot_u_v_contributions

logger = setup_logger()
logger.setLevel(logging.INFO)

SEED = 1423413

REF_EPOCH = 20000
FITNAMES = [
    "250713-01-L0-nnpdf-like",
    "250713-03-L2-nnpdf-like",
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

    PDF_EVOLUTION_DIR = PLOT_DIR / "evolution"
    PDF_EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)

    COVARIANCE_DIR = PLOT_DIR / "covariance"
    COVARIANCE_DIR.mkdir(parents=True, exist_ok=True)

    UV_DIR = PLOT_DIR / "u_v_decomposition"
    UV_DIR.mkdir(parents=True, exist_ok=True)

    DISTANCE_DIR = PLOT_DIR / "distance_from_input"
    DISTANCE_DIR.mkdir(parents=True, exist_ok=True)

    DISTANCE_TR_DIR = PLOT_DIR / "distance_from_training"
    DISTANCE_TR_DIR.mkdir(parents=True, exist_ok=True)

    for fitname in FITNAMES:
        context = FitContext(fitname, force_serialize=parser.parse_args().force)
        datatype = context.get_config("metadata", "arguments")["data"]

        # Evolution with random initialisation, at different epochs
        plot_evolution_from_initialisation(
            context,
            ref_epoch=REF_EPOCH,
            epochs=[700, 5000, 20000],
            colors=[COLOR_TS, COLOR_AS_REF1, COLOR_AS_REF2, COLOR_AS_REF3],
            filename=f"evolution_epochs_700_5000_20000_{datatype}.pdf",
            plot_dir=PDF_EVOLUTION_DIR,
            save_fig=True,
        )

        # Evolution with random initialisation, at the last epoch
        plot_evolution_from_initialisation(
            context,
            ref_epoch=REF_EPOCH,
            epochs=[50000],
            show_true=True,
            plot_dir=PDF_EVOLUTION_DIR,
            save_fig=True,
            colors=[COLOR_TS, COLOR_AS],
            filename=f"evolution_epoch_50000_{datatype}.pdf",
        )

        # Plot of U and V contributions
        epochs_to_plot = [0, 10, 50, 100, 20000]
        for epoch in epochs_to_plot:
            plot_u_v_contributions(
                context,
                ref_epoch=REF_EPOCH,
                ev_epoch=epoch,
                seed=SEED,
                save_fig=True,
                plot_dir=UV_DIR,
                filename=f"evolution_u_v_{epoch}_{datatype}.pdf",
            )

        # Plot of the evolution vs the empirical trained solution
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
                show_true=False,
                save_fig=True,
                plot_dir=PDF_EVOLUTION_DIR,
                filename=f"evolution_vs_trained_epoch_{epoch}_{datatype}.pdf",
            )

        # Distance plot
        epochs_to_plot = [1000, 10000, 20000, 50000]
        for epoch in epochs_to_plot:
            plot_distance_from_input(
                context,
                ref_epoch=20000,
                epoch=epoch,
                seed=SEED,
                show_std=True,
                save_fig=True,
                filename=f"distance_plot_from_input_epoch_{epoch}_{datatype}.pdf",
                plot_dir=DISTANCE_DIR,
                title=rf"$\textrm{{{datatype} data}}$",
            )
            plot_distance_from_train(
                context,
                ref_epoch=20000,
                epoch=epoch,
                seed=SEED,
                show_std=True,
                save_fig=True,
                filename=f"distance_plot_from_training_epoch_{epoch}_{datatype}.pdf",
                plot_dir=DISTANCE_TR_DIR,
                title=rf"$\textrm{{{datatype} data}}$",
            )

        plot_covariance_decomposition(
            context,
            ref_epoch=20000,
            epochs=[0, 1, 100],
            seed=SEED,
            save_fig=True,
            plot_dir=COVARIANCE_DIR,
        )

        plot_cov_compare_tr_an(
            context,
            ref_epoch=20000,
            epochs=[0, 500, 1000, 10000],
            seed=SEED,
            save_fig=True,
            plot_dir=COVARIANCE_DIR,
        )

        # Plot the diagonal error decomposition
        epochs = [0, 100, 500, 1000, 10000, 20000]
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
                ax_specs={
                    "set_yscale": "linear",
                    "set_xticks": [1, 10, 20, 30, 40, 50],
                },  # Convert range to list
                title=rf"$\textrm{{Decomposition at }} T = {epoch}$",
                xlabel=r"$x-{\rm grids}$",
                ylabel=r"$\sigma~(\sqrt{C_{ii}})$",
                save_fig=True,
                plot_dir=COVARIANCE_DIR,
                filename=f"diag_error_decomposition_epoch_{epoch}_{datatype}.pdf",
            )


if __name__ == "__main__":
    main()
