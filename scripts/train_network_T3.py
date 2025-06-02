"""
This script trains a neural network and saves the evolution in pickle format.
"""

from argparse import ArgumentParser
from datetime import datetime
import importlib.resources as pkg_resources
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dlt import data
from dlt.log import MyHandler
from dlt.model import PDFmodel, generate_mse_loss

data_path = Path(pkg_resources.files(data) / "BCDMS_data")

log = logging.getLogger()
log.addHandler(MyHandler())


def save_metadata(args, save_dir):
    """Save training metadata to a YAML file"""

    # Prepare metadata dictionary
    metadata = {
        "training_info": {
            "timestamp": datetime.now().isoformat(),
            "fit_directory": str(save_dir),
        },
        "arguments": {
            "seed": int(args.seed),
            "data": args.data,
            "tolerance": args.tolerance,
            "max_iterations": int(args.max_iterations),
            "learning_rate": args.learning_rate,
            "callback_rate": args.callback_rate,
            "optimizer": args.optimizer,
            "log_level": args.log_level,
            "profiler": args.profiler,
        },
        "model_info": {
            "architecture": args.layers,
            "activations": args.activation,
            "kernel_initializer": "GlorotNormal",
            "dense_layer": "Dense",
            "outputs": 1,
        },
        "data_info": {
            "data_path": str(data_path),
            "datasets": ["BCDMS"],
        },
    }

    # Save to YAML file
    metadata_file = save_dir / "metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    log.info(f"Metadata saved to: {metadata_file}")
    return metadata


def parse_args():
    parser = ArgumentParser()

    # Add option to load from YAML file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (overrides other arguments)",
    )

    # Positional arguments
    parser.add_argument("replica", help="Replica number")
    parser.add_argument("seed", help="Seed number")

    # Optional arguments
    parser.add_argument(
        "--savedir",
        type=str,
        default=Path(".") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Directory to save the model",
    )
    parser.add_argument(
        "--data",
        default="real",
        help="Data generation method: real (default), L0, L1, L2",
        choices=["real", "L0", "L1", "L2"],
    )
    parser.add_argument(
        "--tolerance", "-t", type=float, default=0.0, help="Tolerance for convergence"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=1e6, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--callback_rate",
        "-r",
        type=int,
        default=100,
        help="Callback rate for the optimizer",
    )
    parser.add_argument(
        "--optimizer",
        default="SGD",
        help="Optimizer to use: SGD (default), Adam",
        choices=["SGD", "Adam"],
    )
    parser.add_argument(
        "--log_level",
        "-l",
        default="INFO",
        help="Set logging level (DEBUG, INFO, WARNING (default), ERROR, CRITICAL)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--profiler", action="store_true", help="Enable memory profiler", default=False
    )
    parser.add_argument(
        "--layers",
        help="Architecture of the network",
        type=int,
        nargs="+",
        default=[28, 20],
    )
    parser.add_argument(
        "--activation", help="Activation function", type=str, default="tanh"
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # Override args with values from YAML
        if "arguments" in config:
            for key, value in config["arguments"].items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"{key} not in args")
                    raise RuntimeError("Unexpected error.")

    return args


def main():
    log.info("Starting training script")

    # Parse arguments
    args = parse_args()

    log.setLevel(getattr(logging, args.log_level))

    # Activate profiler if requested
    if args.profiler:
        log.info("Starting memory profiler")
        import tracemalloc

        tracemalloc.start()

    # Make save directories
    if args.config:
        root_dir = Path(args.config.removesuffix(".yaml"))
    else:
        root_dir = args.savedir

    replica_save_dir = root_dir / f"replica_{str(args.replica)}"
    replica_save_dir.mkdir(parents=True, exist_ok=True)
    log.debug(f"Saving directory: {replica_save_dir}")

    # Save metadata
    if int(args.replica) == 1:
        log.info("Saving metadata")
        save_metadata(args, root_dir)

    # Collect Tommaso's data
    fk_grid = np.load(data_path / "fk_grid.npy")
    data = np.load(data_path / "data.npy")
    FK = np.load(data_path / "FK.npy")
    f_bcdms = np.load(data_path / "f_bcdms.npy")
    Cy = np.load(data_path / "Cy.npy")
    # noise = np.load(data_path / "L1_noise_BCDMS.npy')

    # Prepare index for covariance matrix
    arrays = [
        ["T3" for _ in range(Cy.shape[0])],
        ["BCDMS" for _ in range(Cy.shape[0])],
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=("group", "dataset"))
    Cinv = pd.DataFrame(np.linalg.inv(Cy), index=multi_index, columns=multi_index)
    replica_seed = int(args.seed) + int(args.replica)

    rng_replica = np.random.default_rng(replica_seed)
    rng_l1 = np.random.default_rng(int(args.seed))

    if args.data == "real":
        L = np.linalg.cholesky(Cy)
        y = data + rng_replica.normal(size=(Cy.shape[0])) @ L
        log.info(f"Using real data with seed {replica_seed}")

    elif args.data in ["L0", "L1", "L2"]:
        y = FK @ f_bcdms
        log.info(f"L0 data generated")

        if args.data == "L1" or args.data == "L2":
            L = np.linalg.cholesky(Cy)
            y_l1 = FK @ f_bcdms
            y = y_l1 + rng_l1.normal(size=(Cy.shape[0])) @ L
            log.info(f"L1 data generated with seed {int(args.seed)}")

            if args.data == "L2":
                y = y + rng_replica.normal(size=(Cy.shape[0])) @ L
                log.info(f"L2 data generated with seed {replica_seed}")
    else:
        log.error("Please specify --realdata, --L0, --L1 or --L2")
        raise ValueError()

    # ========== Model ==========
    mse_loss = generate_mse_loss(Cinv)
    pdf = PDFmodel(
        dense_layer="Dense",
        input=fk_grid,
        outputs=1,
        architecture=args.layers,
        activations=[args.activation for _ in range(len(args.layers))],
        kernel_initializer="GlorotNormal",
        user_ki_args=None,
        seed=replica_seed,
    )
    pdf.model.summary()

    central_data_dict = {"BCDMS": y}
    fk_dict = {"BCDMS": FK}

    # Train the network
    log.info(f"Chi2 tolerance: {args.tolerance}")
    log.info(f"Maximum iterations: {int(args.max_iterations)}")
    log.info(f"Learning rate: {args.learning_rate}")
    log.info(f"Callback rate: {args.callback_rate}")
    log.info("Starting training...")
    pdf.train_network_gd(
        data=central_data_dict,
        FK_dict=fk_dict,
        loss_func=mse_loss,
        learning_rate=args.learning_rate,
        tol=args.tolerance,
        logging=True,
        callback=True,
        max_epochs=int(args.max_iterations),
        log_fr=args.callback_rate,
        savedir=replica_save_dir,
        optimizer=args.optimizer,
    )

    if args.profiler:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        print("[ Top 10 Memory Consuming Lines ]")
        for stat in top_stats[:10]:
            print(stat)


if __name__ == "__main__":
    main()
