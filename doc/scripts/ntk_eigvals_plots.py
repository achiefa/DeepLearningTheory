#!/usr/bin/env python3
"""This script produces a comparison of the following quantities:
1. Eigenvalues of the NTK
2. Relative eigenvalues of the NTK
3. Delta NTK
4. Relative delta NTK
"""
import logging

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

from argparse import ArgumentParser

import yaml

from yadlt.context import FitContext
from yadlt.distribution import combine_distributions
from yadlt.log import setup_logger
from yadlt.plotting import produce_pdf_plot

logger = setup_logger()
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Config file",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="ntk_eigvals.pdf",
        help="Filename to save the plot.",
    )
    args = parser.parse_args()

    if args.plot_dir is not None:
        from yadlt.plotting import set_plot_dir

        set_plot_dir(args.plot_dir)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    fitnames = config["fitnames"]
    fitlabels = config.get("fitlabels", None)
    fitcolors = config.get("fitcolors", None)
    eigvals = config.get("eigvals", [0])

    eigvals_by_fit = []
    epochs = None

    for fitname in fitnames:
        context = FitContext(fitname)
        if epochs is None:
            epochs = context.get_config("replicas", "common_epochs")
        else:
            assert epochs == context.get_config(
                "replicas", "common_epochs"
            ), "Epochs do not match across fits."
        tmp = combine_distributions(context.eigvals_time)
        tmp.set_name(fitname)
        eigvals_by_fit.append(tmp)

    for idx_eig in eigvals:
        eigvals_grids = []
        for eigval in eigvals_by_fit:
            sliced_distribution = eigval.slice((slice(None), idx_eig - 1))
            eigvals_grids.append(sliced_distribution)

        filename = f"{args.filename}_n_{idx_eig}.pdf"

        produce_pdf_plot(
            epochs,
            eigvals_grids,
            normalize_to=0,
            xlabel=r"${\rm Epoch}$",
            ylabel=r"$\lambda^{(" + str(idx_eig) + ")}$",
            ratio_label=r"$\delta \lambda^{(" + str(idx_eig) + ")}$",
            labels=fitlabels,
            colors=fitcolors,
            save_fig=True,
            filename=filename,
        )


if __name__ == "__main__":
    main()
