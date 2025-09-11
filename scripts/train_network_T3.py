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


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        default=None,
        help="Path to YAML configuration file (overrides other arguments)",
    )
    parser.add_argument("replica", help="Replica number")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    log.info("Starting training script")

    log.setLevel(getattr(logging, config["arguments"].get("log_level", "INFO")))

    # Make save directories
    root_dir = Path(args.config.removesuffix(".yaml"))

    replica_save_dir = root_dir / f"replica_{str(args.replica)}"
    replica_save_dir.mkdir(parents=True, exist_ok=True)
    log.debug(f"Saving directory: {replica_save_dir}")

    # Save metadata
    if int(args.replica) == 1:
        metadata_file = root_dir / "metadata.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        log.info(f"Metadata saved to: {metadata_file}")

    # Collect Tommaso's data
    fk_grid = load_bcdms_grid()
    data = load_bcdms_data()
    FK = load_bcdms_fk()
    f_bcdms = load_bcdms_pdf()
    Cy = load_bcdms_cov()

    # Generate data
    replica_seed = int(config["arguments"]["seed"]) + int(args.replica)
    rng_replica = np.random.default_rng(replica_seed)
    rng_l1 = np.random.default_rng(int(config["arguments"]["seed"]))

    data_type = config["arguments"]["data"]
    if data_type == "real":
        L = np.linalg.cholesky(Cy)
        y = data + rng_replica.normal(size=(Cy.shape[0])) @ L
        log.info(f"Using real data with seed {replica_seed}")

    elif data_type in ["L0", "L1", "L2"]:
        y = FK @ f_bcdms
        log.info(f"L0 data generated")

        if data_type == "L1" or data_type == "L2":
            L = np.linalg.cholesky(Cy)
            y_l1 = FK @ f_bcdms
            y = y_l1 + rng_l1.normal(size=(Cy.shape[0])) @ L
            log.info(f"L1 data generated with seed {int(config['arguments']['seed'])}")

            if data_type == "L2":
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
    arch = config["model_info"]["architecture"]
    activation = config["model_info"]["activation"]
    use_scaled_input = config["model_info"].get("use_scaled_input", False)
    use_preprocessing = config["model_info"].get("use_preprocessing", False)
    log.info("Generating PDF model")
    pdf_model = generate_pdf_model(
        outputs=1,
        architecture=arch,
        activations=[activation for _ in range(len(arch))],
        kernel_initializer="GlorotNormal",
        bias_initializer="zeros",
        user_ki_args=None,
        seed=replica_seed,
        scaled_input=use_scaled_input,
        preprocessing=use_preprocessing,
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
    optimizer = config["arguments"]["optimizer"]
    learning_rate = config["arguments"]["learning_rate"]
    cliponorm = config["arguments"].get("clipnorm", None)
    if optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=float(learning_rate), clipnorm=cliponorm
        )
    elif optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    train_model.compile(optimizer=optimizer, loss=[chi2])

    # Train the model
    callback_freq = config["arguments"]["callback_freq"]
    x = tf.constant(fk_grid.reshape(1, -1, 1), dtype=tf.float32)
    log_cb = LoggingCallback(log_frequency=callback_freq, ndata=data.size)
    save_cb = WeightStorageCallback(
        storage_frequency=callback_freq,
        storage_path=replica_save_dir,
        training_data=(x, y.reshape(1, -1)),
    )
    nan_cb = NaNCallback()

    max_iter = config["arguments"]["max_iterations"]
    _ = train_model.fit(
        x,
        y.reshape(1, -1),
        epochs=int(max_iter),
        verbose=0,
        callbacks=[log_cb, save_cb, nan_cb],
    )

    log.info("Training completed")
    log.info(f"Model saved to {replica_save_dir}")
    log.info("Saving training history")
    save_cb.save_history()


if __name__ == "__main__":
    main()
