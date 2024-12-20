"""
This script trains a neural network and saves the evolution in pickle format.
"""

import sys

# Add the path to the library folder
sys.path.append('./lib')

from utils import XGRID
from model import PDFmodel, generate_mse_loss
from gen_dicts import generate_dicts
from validphys.api import API

import numpy as np
import pandas as pd
import pickle
from datetime import date


# List of DIS dataset
dataset_inputs = [
  #{'dataset': 'NMC_NC_NOTFIXED_DW_EM-F2', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'NMC_NC_NOTFIXED_P_EM-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'SLAC_NC_NOTFIXED_P_DW_EM-F2', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'SLAC_NC_NOTFIXED_D_DW_EM-F2', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'BCDMS_NC_NOTFIXED_P_DW_EM-F2', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'BCDMS_NC_NOTFIXED_D_DW_EM-F2', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED', 'cfac': ['MAS'], 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED', 'cfac': ['MAS'], 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_318GEV_EM-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_225GEV_EP-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_251GEV_EP-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_300GEV_EP-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_318GEV_EP-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_CC_318GEV_EM-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_CC_318GEV_EP-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_318GEV_EAVG_CHARM-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
  {'dataset': 'HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED', 'frac': 0.75, 'variant': 'legacy'},
]

# Dictionary for validphys API
common_dict = dict(
    dataset_inputs=dataset_inputs,
    metadata_group="nnpdf31_process",
    use_cuts='internal',
    datacuts={'q2min': 3.49, 'w2min': 12.5},
    theoryid=40000000,
    t0pdfset='NNPDF40_nnlo_as_01180',
    use_t0=True
)


if __name__ == "__main__":
  # Create the model
  pdf = PDFmodel(input=XGRID,
               outputs=9,
               architecture=[25,28],
               activations=['tanh', 'tanh'],
               kernel_initializer='RandomNormal',
               user_ki_args={'mean': 0.0, 'stddev': 1.0},
               seed=1)

  learning_rate = 1.e-7

  groups_data = API.procs_data(**common_dict)
  tuple_of_dicts = generate_dicts(groups_data)
  fk_table_dict = tuple_of_dicts.fk_tables
  central_data_dict = tuple_of_dicts.central_data

  C_sys = API.dataset_inputs_t0_covmat_from_systematics(**common_dict)
  C = API.groups_covmat_no_table(**common_dict)
  C_index = C.index
  C_col = C.columns
  Cinv = np.linalg.inv(C)
  Cinv = pd.DataFrame(Cinv, index=C_index, columns=C_col)

  # Initialize the loss function
  mse_loss = generate_mse_loss(Cinv)

  # Train the network
  model_time, steps = pdf.train_network_gd(data=central_data_dict,
                                         FK_dict=fk_table_dict,
                                         loss_func=mse_loss,
                                         learning_rate=learning_rate,
                                         tol=1e-8,
                                         logging=True,
                                         callback=True)

today_date = date.today().timetuple()

with open(f'training_{today_date[0]}_{today_date[1]}_{today_date[2]}.pkl', 'wb') as file:
  pickle.dump((model_time, steps), file)
