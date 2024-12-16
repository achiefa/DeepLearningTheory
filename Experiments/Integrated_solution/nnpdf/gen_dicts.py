from validphys.pineparser import pineappl_reader
from n3fit.layers.observable import compute_float_mask
from n3fit.backends import operations as op
from n3fit.layers import DIS

from collections import defaultdict, namedtuple
import numpy as np

from utils import XGRID


def generate_dicts(groups_data):

  # Initialize named tuple
  result_tuple = namedtuple('result_tuple', 
                            ('fk_tables',
                             'central_data',
                             'padded_fk',
                             'xgrid_mask'))

  # Initialise the dictionaries
  fk_table_dict = defaultdict(list)
  central_data_dict = {}
  padded_fk_dict = defaultdict(list)
  xgrid_masks_dict = defaultdict(list)

  total_ndata_wc = 0
  for idx_proc, group_proc in enumerate(groups_data):
    for idx_exp, exp_set in enumerate(group_proc.datasets):
    
      dataset_name = exp_set.name
      dataset_size = exp_set.load_commondata().ndata
      total_ndata_wc += dataset_size

      # Collect FKSpecs and cuts
      fkspecs = exp_set.fkspecs
      cuts = exp_set.cuts

      # Read FKData and FK table in numpy version
      fk_data = pineappl_reader(fkspecs[0]).with_cuts(cuts)
      fk_table = fk_data.get_np_fktable()

      # xgrid for this dataset
      xgrid = fk_data.xgrid

      # Check that XGRID is just a small-x extension
      # of xgrid
      res = True
      for i, x in enumerate(xgrid):
        offset = 50 - xgrid.size
        try:
          assert(np.isclose(x, XGRID[offset+i]))
        except AssertionError:
          print(f"XGRID is not an extension for {dataset_name}.")

      # Load DIS object for padding the FK table
      dis = DIS(
        [fk_data],
        [fk_table],
        dataset_name,
        None,
        exp_set.op,
        n_replicas=1,
        name=f"dat_{dataset_name}"
      )
      
      # OLD
      # Pad the fk table so that (N, x, 9) -> (N, x, 14)
      mask = dis.masks[0]
      padded_fk_table = dis.fktables[0]#dis.pad_fk(dis.fktables[0], mask)
      padded_fk_dict[dataset_name] = dis.pad_fk(dis.fktables[0], mask)

      # Extend xgrid to low-x (N, x, 14) -> (N, 50, 14)
      xgrid_mask = np.zeros(XGRID.size, dtype=bool)
      offset = XGRID.size - xgrid.size
      for i in range(xgrid.size):
        xgrid_mask[offset + i] = True
      xgrid_mask = compute_float_mask(xgrid_mask)
      paddedx_fk_table = op.einsum('Xx, nFx -> nXF', xgrid_mask, padded_fk_table)
      xgrid_masks_dict[dataset_name] = xgrid_mask
      # Check the mask in x is applied correctly
      #for i in range(XGRID.size):
      if i >= offset:
        try:
          assert(np.allclose(paddedx_fk_table[:,i,:], padded_fk_table[:,:,i - offset]))
        except AssertionError:
          print(f'Problem in the unchanged values for {dataset_name}')
      else:
        try:
          assert(np.allclose(paddedx_fk_table[:,i,:], np.zeros_like(paddedx_fk_table[:,i,:])))
        except AssertionError:
          print(f'Problem in the extension for {dataset_name}')

      # Save to dict
      fk_table_dict[dataset_name] = paddedx_fk_table
      central_data_dict[dataset_name] = exp_set.load_commondata().with_cuts(cuts).central_values.to_numpy()

  res = result_tuple(fk_tables=fk_table_dict,
               central_data=central_data_dict,
               padded_fk=padded_fk_dict,
               xgrid_mask=xgrid_masks_dict)
  return res


