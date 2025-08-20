#!/usr/bin/env python3
"""This script produces a comparison of the following quantities:
1. Eigenvalues of the NTK
2. Relative eigenvalues of the NTK
3. Delta NTK
4. Relative delta NTK
"""
from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.context import FitContext
from yadlt.log import setup_logger
from yadlt.plotting.plot_alignment import plot_theta_m_alignment, produce_alignment_plot

logger = setup_logger()
logger.setLevel(logging.INFO)

FITNAME = "250713-03-L2-nnpdf-like"
REF_REPLICA = 21
REF_EPOCHS = [0, 1000, 20000]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    args = parser.parse_args()
    PLOT_DIR = Path(args.plot_dir) if args.plot_dir else Path(__file__).parent / "plots"
    NTK_PLOT_DIR = PLOT_DIR / "ntk_pheno"
    NTK_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    context = FitContext(FITNAME)
    datatype = context.get_config("metadata", "arguments")["data"]
    plot_theta_m_alignment(
        context=context,
        replica=REF_REPLICA,
        epochs=REF_EPOCHS,
        filename=f"ntk_alignment_{datatype}.pdf",
        save_fig=True,
        plot_dir=NTK_PLOT_DIR,
        cut_noise=20,
    )

    produce_alignment_plot(
        context,
        [
            0,
            1,
            2,
            3,
            4,
        ],
        save_fig=True,
        plot_dir=NTK_PLOT_DIR,
        filename=f"ntk_align_fin_1_{datatype}.pdf",
    )
    produce_alignment_plot(
        context,
        [5, 6, 7, 8, 9],
        save_fig=True,
        plot_dir=NTK_PLOT_DIR,
        filename=f"ntk_align_fin_2_{datatype}.pdf",
    )


if __name__ == "__main__":
    main()
