"""
This script trains a neural network and saves the evolution in pickle format.
"""

from argparse import ArgumentParser
from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from yadlt.callback import LoggingCallback, NaNCallback, WeightStorageCallback
from yadlt.layers import Convolution
from yadlt.load_data import (
    load_bcdms_cov,
    load_bcdms_data,
    load_bcdms_fk,
    load_bcdms_grid,
    load_bcdms_pdf,
)
from yadlt.log import MyHandler
from yadlt.model import Chi2, generate_pdf_model

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
            "max_iterations": int(args.max_iterations),
            "learning_rate": args.learning_rate,
            "callback_freq": args.callback_freq,
            "optimizer": args.optimizer,
            "log_level": args.log_level,
        },
        "model_info": {
            "architecture": args.architecture,
            "activations": args.activation,
            "kernel_initializer": "GlorotNormal",
            "outputs": 1,
            "use_scaled_input": args.use_scaled_input,
            "use_preprocessing": args.use_preprocessing,
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

    # Optional arguments
    parser.add_argument("--seed", help="Seed number", default=57298, type=int)
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
        "--max_iterations", type=int, default=1e6, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--callback_freq",
        "-r",
        type=int,
        default=100,
        help="Callback frequency",
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
        "--architecture",
        help="Architecture of the network",
        type=int,
        nargs="+",
        default=[28, 20],
    )
    parser.add_argument(
        "--activation", help="Activation function", type=str, default="tanh"
    )
    parser.add_argument(
        "--use_scaled_input",
        action="store_true",
        help="Use scaled input (default: False)",
    )
    parser.add_argument(
        "--use_preprocessing",
        action="store_true",
        help="Use preprocessing (default: False)",
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

        # Model info
        if "model_info" in config:
            for key, value in config["model_info"].items():
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

    # Make save directories
    if args.config:
        root_dir = Path(args.config.removesuffix(".yaml"))
    else:
        root_dir = Path(args.savedir)

    replica_save_dir = root_dir / f"replica_{str(args.replica)}"
    replica_save_dir.mkdir(parents=True, exist_ok=True)
    log.debug(f"Saving directory: {replica_save_dir}")

    # Save metadata
    if int(args.replica) == 1:
        log.info("Saving metadata")
        save_metadata(args, root_dir)

    # Collect Tommaso's data
    fk_grid = load_bcdms_grid()
    data = load_bcdms_data()
    FK = load_bcdms_fk()
    f_bcdms = load_bcdms_pdf()
    Cy = load_bcdms_cov()

    # Generate data
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

    # Save the data
    np.save(replica_save_dir / "data.npy", y)

    # Prepare the loss function
    chi2 = Chi2(Cy)

    # Prepare the PDF model
    log.info("Generating PDF model")
    pdf_model = generate_pdf_model(
        outputs=1,
        architecture=args.architecture,
        activations=[args.activation for _ in range(len(args.architecture))],
        kernel_initializer="GlorotNormal",
        bias_initializer="zeros",
        user_ki_args=None,
        seed=replica_seed,
        scaled_input=args.use_scaled_input,
        preprocessing=args.use_preprocessing,
    )
    pdf_model.summary()
    pdf_model.get_layer("pdf_raw").summary()
    model_input = pdf_model.input

    # Prepare convolution layer
    FK = FK.reshape(FK.shape[0], 1, FK.shape[1])
    convolution = Convolution(FK, basis=[0], nfl=1, name="BCDMS_convolution")

    # Construct observable model
    obs = tf.keras.models.Sequential([pdf_model, convolution], name="Observable")

    # Define trainable model
    train_model = tf.keras.models.Model(inputs=model_input, outputs=obs(model_input))
    if args.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=float(args.learning_rate), clipnorm=1.0
        )
    elif args.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    train_model.compile(optimizer=optimizer, loss=[chi2])

    # Train the model
    x = tf.constant(fk_grid.reshape(1, -1, 1), dtype=tf.float32)
    log_cb = LoggingCallback(log_frequency=args.callback_freq, ndata=data.size)
    save_cb = WeightStorageCallback(
        storage_frequency=args.callback_freq,
        storage_path=replica_save_dir,
        training_data=(x, y.reshape(1, -1)),
    )
    nan_cb = NaNCallback()

    _ = train_model.fit(
        x,
        y.reshape(1, -1),
        epochs=int(args.max_iterations),
        verbose=0,
        callbacks=[log_cb, save_cb, nan_cb],
    )

    log.info("Training completed")
    log.info(f"Model saved to {replica_save_dir}")
    log.info("Saving training history")
    save_cb.save_history()


if __name__ == "__main__":
    main()
