from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.log import setup_logger
from yadlt.plotting.plot_delta_ntk import plot_delta_ntk

logger = setup_logger()
logger.setLevel(logging.INFO)

DELTA_NTK_DATA_DICT = {
    "fitnames": [
        "250713-01-L0-nnpdf-like",
        "250713-02-L1-nnpdf-like",
        "250713-03-L2-nnpdf-like",
    ],
    "fitlabels": ["$\\textrm{L0}$", "$\\textrm{L1}$", "$\\textrm{L2}$"],
    "fitcolors": ["C0", "C1", "C2"],
    "filename": "delta_ntk.pdf",
}

DELTA_NTK_ARCH_DICT = {
    "fitnames": ["250713-02-L1-nnpdf-like", "250713-06-L1-large"],
    "fitlabels": ["$\\textrm{Arch. } [28,20]$", "$\\textrm{Arch. } [100,100]$"],
    "fitcolors": ["C0", "C1"],
    "filename": "delta_ntk_arch.pdf",
}


def main():
    parser = ArgumentParser(description="Compute and plot PDFs.")
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

    plot_delta_ntk(**DELTA_NTK_DATA_DICT, save_fig=True, plot_dir=NTK_PLOT_DIR)
    plot_delta_ntk(**DELTA_NTK_ARCH_DICT, save_fig=True, plot_dir=NTK_PLOT_DIR)


if __name__ == "__main__":
    main()
