#!/usr/bin/env python3
"""This script produces the plots for the kernel recursion"""

from argparse import ArgumentParser
import logging
from pathlib import Path

from yadlt import load_data
from yadlt.log import setup_logger
from yadlt.model import generate_pdf_model
from yadlt.plotting.plot_kernel_recursion import plot_kernel_at_layer

logger = setup_logger()
logger.setLevel(logging.INFO)


# Load Tommaso's file
fk_grid = load_data.load_bcdms_grid()

SEED = 231231233
NREPLICAS = 100
SAMPLES = 10000
BATCH_SIZE = 1000

ARCHITECTURES = [
    [10, 10],
    [25, 20],
    [100, 100],
    [500, 500],
    # [1000],
    # [2000, 2000],
    # [10000, 10000],
]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--nreplicas",
        "-n",
        type=int,
        default=NREPLICAS,
        help="Number of replicas to compute the NTK",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=SEED,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--plot-dir",
        type=None,
        default=None,
        help="Directory to save the plots. If not specified, uses the default plot directory.",
    )
    args = parser.parse_args()
    DIR = Path(args.plot_dir) if args.plot_dir else Path(__file__).parent / "plots"
    PLOT_DIR = DIR / "kernel_recursion"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for architecture in ARCHITECTURES:
        logger.info(f"Computing kernel for architecture: {architecture}")
        savedir = (
            PLOT_DIR
            / f'kernel_recursion_{str(architecture).replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")}'
        )
        savedir.mkdir(parents=True, exist_ok=True)

        model_ens = []
        for rep in range(args.nreplicas):
            model = generate_pdf_model(
                outputs=1,
                architecture=architecture,
                activations=["tanh" for _ in architecture],
                kernel_initializer="GlorotNormal",
                user_ki_args=None,
                seed=args.seed + rep,
                scaled_input=False,
                preprocessing=False,
            )
            model_ens.append(model)

        # For input layer
        plot_kwargs_list = [
            {"save_fig": True, "plot_dir": savedir, "filename": f"k_input_layer.pdf"},
            {},
        ]
        plot_kernel_at_layer(
            model_ens,
            layer_idx=0,
            input_data=fk_grid,
            num_samples=SAMPLES,
            batch_size=BATCH_SIZE,
            diag_idx=0,
            offdiag_idx=None,
            plot_kwargs_list=plot_kwargs_list,
        )

        # First deep layer
        plot_kwargs_list = [
            {
                "save_fig": True,
                "plot_dir": savedir,
                "filename": f"k_first_layer_diag.pdf",
                "text_dict": {
                    "x": -0.3,
                    "y": 0.5,
                    "s": r"$\textrm{Diagonal}$" + "\n" + r"$\rm{elements}$",
                },
            },
            {
                "save_fig": True,
                "plot_dir": savedir,
                "filename": f"k_first_layer_offdiag.pdf",
                "text_dict": {
                    "x": -0.3,
                    "y": 0.5,
                    "s": r"$\textrm{Off-diagonal}$" + "\n" + r"$\rm{elements}$",
                },
            },
        ]
        plot_kernel_at_layer(
            model_ens,
            layer_idx=1,
            input_data=fk_grid,
            num_samples=SAMPLES,
            batch_size=BATCH_SIZE,
            diag_idx=0,
            offdiag_idx=architecture[0] // 2,
            plot_kwargs_list=plot_kwargs_list,
        )

        # Second deep layer
        plot_kwargs_list = [
            {
                "save_fig": True,
                "plot_dir": savedir,
                "filename": f"k_second_layer_diag.pdf",
                "text_dict": {
                    "x": -0.3,
                    "y": 0.5,
                    "s": r"$\textrm{Diagonal}$" + "\n" + r"$\rm{elements}$",
                },
            },
            {
                "save_fig": True,
                "plot_dir": savedir,
                "filename": f"k_second_layer_offdiag.pdf",
                "text_dict": {
                    "x": -0.3,
                    "y": 0.5,
                    "s": r"$\textrm{Off-diagonal}$" + "\n" + r"$\rm{elements}$",
                },
            },
        ]
        plot_kernel_at_layer(
            model_ens,
            layer_idx=2,
            input_data=fk_grid,
            num_samples=SAMPLES,
            batch_size=BATCH_SIZE,
            diag_idx=0,
            offdiag_idx=architecture[0] // 2,
            plot_kwargs_list=plot_kwargs_list,
        )


if __name__ == "__main__":
    main()
