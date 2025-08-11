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
from yadlt.plotting.plot_alignment_matrix import plot_alignment_matrix

logger = setup_logger()
logger.setLevel(logging.INFO)

FITNAME = "250713-03-L2-nnpdf-like"
REF_REPLICA = 21
REF_EPOCHS = [0, 1000, 20000]
FILENAME = "ntk_alignment.pdf"


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
    plot_alignment_matrix(
        context=context,
        replica=REF_REPLICA,
        epochs=REF_EPOCHS,
        filename=FILENAME,
        save_fig=True,
        plot_dir=NTK_PLOT_DIR,
    )


if __name__ == "__main__":
    main()
