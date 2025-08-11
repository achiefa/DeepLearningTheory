from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.context import FitContext
from yadlt.log import setup_logger
from yadlt.plotting.plot_distance import plot_distance
from yadlt.plotting.plot_evolution_pdf import plot_evolution_from_initialisation
from yadlt.plotting.plot_u_v_contribution import plot_u_v_contributions

logger = setup_logger()
logger.setLevel(logging.INFO)

SEED = 1423413

REF_EPOCH = 20000
FITNAMES = [
    "250713-01-L0-nnpdf-like",
    "250713-03-L2-nnpdf-like",
]


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
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for fitname in FITNAMES:
        context = FitContext(fitname, force_serialize=parser.parse_args().force)

        # Evolution with random initialisation, at different epochs
        plot_evolution_from_initialisation(
            context,
            ref_epoch=REF_EPOCH,
            epochs=[700, 5000, 50000],
            filename="init_epochs",
            plot_dir=PLOT_DIR,
            save_fig=True,
        )
        # Evolution with random initialisation, at the last epoch
        plot_evolution_from_initialisation(
            context,
            ref_epoch=REF_EPOCH,
            epochs=[-1],
            filename="init_last_epoch",
            show_true=True,
            plot_dir=PLOT_DIR,
            save_fig=True,
        )
        plot_u_v_contributions(
            context,
            ref_epoch=REF_EPOCH,
            ev_epoch=0,
            seed=SEED,
            save_fig=True,
            plot_dir=PLOT_DIR,
        )
        plot_u_v_contributions(
            context,
            ref_epoch=REF_EPOCH,
            ev_epoch=10,
            seed=SEED,
            save_fig=True,
            plot_dir=PLOT_DIR,
        )
        plot_u_v_contributions(
            context,
            ref_epoch=REF_EPOCH,
            ev_epoch=50,
            seed=SEED,
            save_fig=True,
            plot_dir=PLOT_DIR,
        )
        plot_u_v_contributions(
            context,
            ref_epoch=REF_EPOCH,
            ev_epoch=100,
            seed=SEED,
            save_fig=True,
            plot_dir=PLOT_DIR,
        )
        plot_u_v_contributions(
            context,
            ref_epoch=REF_EPOCH,
            ev_epoch=50000,
            seed=SEED,
            save_fig=True,
            plot_dir=PLOT_DIR,
        )
        plot_distance(
            context, ref_epoch=REF_EPOCH, seed=SEED, save_fig=True, plot_dir=PLOT_DIR
        )


if __name__ == "__main__":
    main()
