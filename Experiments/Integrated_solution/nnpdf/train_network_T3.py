"""
This script trains a neural network and saves the evolution in pickle format.
"""
import sys
sys.path.append('./lib')

import logging
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from log import MyHandler
from model import PDFmodel, generate_mse_loss

log = logging.getLogger()
log.addHandler(MyHandler())

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('replica', help='Replica number')
  parser.add_argument('seed', help='Seed number')
  parser.add_argument('--savedir', type=str, default=None, help='Directory to save the model')
  parser.add_argument('--data',
                      default='real', 
                      help='Data generation method: real (default), L1, L2',
                      choices=['real', 'L1', 'L2'])
  parser.add_argument('--tol', type=float, default=0.0, help='Tolerance for convergence')
  parser.add_argument('--max_iter', type=int, default=1e6, help='Maximum number of iterations')
  parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
  parser.add_argument('--cb_rate', type=int, default=100, help='Callback rate for the optimizer')
  parser.add_argument('--optimizer',
                      default='SGD',
                      help='Optimizer to use: SGD (default), Adam',
                      choices=['SGD', 'Adam'])
  parser.add_argument('-l', '--log', 
                      default='INFO', 
                      help='Set logging level (DEBUG, INFO, WARNING (default), ERROR, CRITICAL)',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
  parser.add_argument('--profiler', 
                      action='store_true', 
                      help='Enable memory profiler',
                      default=False)
  args = parser.parse_args()

  if args.profiler:
    log.info('Starting memory profiler')
    import tracemalloc
    tracemalloc.start()

  log.setLevel(getattr(logging, args.log))

  log.info('Starting training script')

  if args.savedir is not None:
    SAVE_DIR = Path(args.savedir) / f'replica_{str(args.replica)}' 
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
  else:
    SAVE_DIR = Path('.') / datetime.now().strftime("%Y-%m-%d_%H-%M-%S") / f'replica_{str(args.replica)}'
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
  log.debug(f'Saving directory: {SAVE_DIR}')

  # Collect Tommaso's data
  fk_grid = np.load('Tommaso/fk_grid.npy')
  data = np.load('Tommaso/data.npy')
  FK = np.load('Tommaso/FK.npy')
  f_bcdms = np.load('Tommaso/f_bcdms.npy')
  Cy = np.load('Tommaso/Cy.npy')
  #noise = np.load('Tommaso/L1_noise_BCDMS.npy')

  # Prepare index for covariance matrix
  arrays = [
    ['T3' for _ in range(Cy.shape[0])],
    ['BCDMS' for _ in range(Cy.shape[0])],
    ]
  multi_index = pd.MultiIndex.from_arrays(arrays, names=('group', 'dataset'))
  Cinv = pd.DataFrame(np.linalg.inv(Cy), index=multi_index, columns=multi_index)
  seed = int(args.seed) + int(args.replica)
  np.random.seed(seed)

  if args.data == 'real':
    L = np.linalg.cholesky(Cy)
    y = data + np.random.normal(size=(Cy.shape[0])) @ L
    log.info(f'Using real data with seed {seed}')
  elif args.data == 'L1':
    y = FK @ f_bcdms
    log.info(f'Using L1 data')
  elif args.data == 'L2':
    L = np.linalg.cholesky(Cy)
    y_l1 = FK @ f_bcdms
    y = y_l1 + np.random.normal(size=(Cy.shape[0])) @ L
    log.info(f'Using L2 data with seed {seed}')
  else:
    log.error('Please specify --realdata, --L1 or --L2')
    raise ValueError()
  
  # ========== Model ==========
  mse_loss = generate_mse_loss(Cinv)
  pdf = PDFmodel(
                 dense_layer='Dense',
                 input=fk_grid,
                 outputs=1,
                 architecture=[28, 25],
                 activations=['tanh', 'tanh'],
                 kernel_initializer='GlorotNormal',
                 user_ki_args=None,
                 seed=seed,)
  pdf.model.summary()

  central_data_dict = {'BCDMS': y}
  fk_dict = {'BCDMS': FK}

  # Train the network
  log.info(f'Chi2 tolerance: {args.tol}')
  log.info(f'Maximum iterations: {int(args.max_iter)}')
  log.info(f'Learning rate: {args.learning_rate}')
  log.info(f'Callback rate: {args.cb_rate}')
  log.info('Starting training...')
  pdf.train_network_gd(data=central_data_dict,
                        FK_dict=fk_dict,
                        loss_func=mse_loss,
                        learning_rate=args.learning_rate,
                        tol=args.tol,
                        logging=True,
                        callback=True,
                        max_epochs=int(args.max_iter),
                        log_fr=args.cb_rate,
                        savedir=SAVE_DIR,
                        optimizer=args.optimizer)
  
  if args.profiler:
    snapshot = tracemalloc.take_snapshot()
    import ipdb; ipdb.set_trace()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 Memory Consuming Lines ]")
    for stat in top_stats[:10]:
        print(stat)