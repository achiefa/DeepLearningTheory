from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt.log import setup_logger
from yadlt.plotting.plot_eigvals_single_plot import plot_eigvals_single_plot

logger = setup_logger()
logger.setLevel(logging.INFO)

### CONFIGURATION FOR PLOTTING
NTK_L0 = {
    "eigval_type": "ntk",
    "fitnames": ["250713-01-L0-nnpdf-like"],
    "group_by": "eigvals",
    "ylabel": "$\\textrm{NTK eigenvalues}$",
    "labels": [
        "$\\lambda^{1}$",
        "$\\lambda^{2}$",
        "$\\lambda^{3}$",
        "$\\lambda^{4}$",
        "$\\lambda^{5}$",
    ],
    "colors": ["C0", "C1", "C2", "C3", "C4"],
    "eigvals": [1, 2, 3, 4, 5],
}

NTK_ARCH_L1 = {
    "eigval_type": "ntk",
    "fitnames": ["250713-02-L1-nnpdf-like", "250713-06-L1-large"],
    "group_by": "fitnames",
    "ylabel": "$\\lambda$",
    "labels": ["$\\textrm{Arch. } [28,20]$", "$\\textrm{Arch. } [100,100]$"],
    "colors": ["C0", "C1"],
    "eigvals": [1, 2, 3],
}

H_L0 = {
    "eigval_type": "h",
    "fitnames": ["250713-01-L0-nnpdf-like"],
    "group_by": "eigvals",
    "ylabel": "$\\textrm{h}$",
    "labels": ["$h^{1}$", "$h^{2}$", "$h^{3}$"],
    "colors": ["C0", "C1", "C2"],
    "eigvals": [1, 2, 3],
}

H_L1 = {
    "eigval_type": "h",
    "fitnames": ["250713-02-L1-nnpdf-like"],
    "group_by": "eigvals",
    "ylabel": "$\\textrm{h}$",
    "labels": ["$h^{1}$", "$h^{2}$", "$h^{3}$"],
    "colors": ["C0", "C1", "C2"],
    "eigvals": [1, 2, 3],
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
    H_PLOT_DIR = PLOT_DIR / "h_matrix"
    NTK_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    H_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Producing NTK eigenvalues plots with L0 data...")
    plot_eigvals_single_plot(
        **NTK_L0,
        filename="ntk_eigvals_single_plot_L0.pdf",
        plot_dir=NTK_PLOT_DIR,
        save_fig=True
    )
    logger.info("Producing NTK eigenvalues plots with different architectures...")
    plot_eigvals_single_plot(
        **NTK_ARCH_L1,
        filename="ntk_eigvals_single_plot_arch.pdf",
        plot_dir=NTK_PLOT_DIR,
        save_fig=True
    )
    logger.info("Producing H eigenvalues plots with L0 data...")
    plot_eigvals_single_plot(
        **H_L0,
        filename="h_eigvals_single_plot_L0.pdf",
        plot_dir=H_PLOT_DIR,
        save_fig=True
    )
    logger.info("Producing H eigenvalues plots with L1 data...")
    plot_eigvals_single_plot(
        **H_L1,
        filename="h_eigvals_single_plot_L1.pdf",
        plot_dir=H_PLOT_DIR,
        save_fig=True
    )


if __name__ == "__main__":
    main()
