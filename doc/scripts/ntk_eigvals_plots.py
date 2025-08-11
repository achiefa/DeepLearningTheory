"""This script produces a comparison of the following quantities:
1. Eigenvalues of the NTK
2. Relative eigenvalues of the NTK
3. Delta NTK
4. Relative delta NTK
"""

from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.log import setup_logger
from yadlt.plotting.plot_eigvals import plot_eigvals

logger = setup_logger()
logger.setLevel(logging.INFO)


NTK_DICT = {
    "fitnames": [
        "250713-01-L0-nnpdf-like",
        "250713-02-L1-nnpdf-like",
        "250713-03-L2-nnpdf-like",
    ],
    "fitlabels": ["$\\textrm{L0}$", "$\\textrm{L1}$", "$\\textrm{L2}$"],
    "fitcolors": ["C0", "C1", "C2"],
    "eigvals": [1, 2, 3, 4, 5],
    "filename_prefix": "ntk_eigvals_L0_L1_L2",
}


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

    plot_eigvals(**NTK_DICT, save_fig=True, plot_dir=NTK_PLOT_DIR)


if __name__ == "__main__":
    main()
