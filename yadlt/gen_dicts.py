from collections import defaultdict, namedtuple

from n3fit.backends import operations as op
from n3fit.layers import DIS
from n3fit.layers.observable import compute_float_mask
import numpy as np
import tensorflow as tf
from utils import XGRID
from validphys.pineparser import pineappl_reader


def generate_dicts(groups_data):

    # Initialize named tuple
    result_tuple = namedtuple(
        "result_tuple", ("fk_tables", "central_data", "padded_fk", "xgrid_mask")
    )

    # Initialise the dictionaries
    fk_table_dict = {}
    central_data_dict = {}
    padded_fk_dict = {}
    xgrid_masks_dict = defaultdict(list)

    large_xgrid = np.array([])

    # Find the largest grid
    for group_proc in groups_data:
        for exp_set in group_proc.datasets:
            fkspecs = exp_set.fkspecs
            cuts = exp_set.cuts
            fk_data = pineappl_reader(fkspecs[0]).with_cuts(cuts)
            xgrid = fk_data.xgrid
            if xgrid.size > large_xgrid.size:
                large_xgrid = xgrid

    total_ndata_wc = 0
    for group_proc in groups_data:
        for exp_set in group_proc.datasets:
            dataset_name = exp_set.name
            dataset_size = exp_set.load_commondata().ndata
            total_ndata_wc += dataset_size

            # Collect FKSpecs and cuts
            fkspecs = exp_set.fkspecs
            cuts = exp_set.cuts

            # Read FKData and FK table in numpy version
            fk_data = pineappl_reader(fkspecs[0]).with_cuts(cuts)
            fk_table = fk_data.get_np_fktable()
            print(fk_table.shape)

            # xgrid for this dataset
            xgrid = fk_data.xgrid

            # Check that `large_xgrid` is just a small-x extension of xgrid
            res = True
            for i, x in enumerate(xgrid):
                offset = large_xgrid.size - xgrid.size
                try:
                    assert np.isclose(x, large_xgrid[offset + i])
                except AssertionError:
                    print(f"`large_xgrid` is not an extension for {dataset_name}.")

            # Load DIS object for padding the FK table
            dis = DIS(
                [fk_data],
                [fk_table],
                dataset_name,
                None,
                exp_set.op,
                n_replicas=1,
                name=f"dat_{dataset_name}",
            )

            # This cast is needed if we want float64 fk tables. The reason being
            # that this mask is then applied to the padded fk tables, which are
            # float64. However, if we kept the mask float32, then the final padded
            # fk tables would be converted to float32 as well.
            mask = tf.cast(dis.masks[0], dtype=tf.float64)
            # fk_table = dis.fktables[0]
            # if not np.allclose(fk_table, fk_table):
            #  print(f'Problem in the fk table for {dataset_name}')

            # Pad the fk table so that (N, x, 9) -> (N, x, 14)
            # Combine an fk table and a mask into an fk table padded with zeroes
            # for the inactive flavours, to be contracted with the full PDF.
            padded_fk_dict[dataset_name] = dis.pad_fk(dis.fktables[0], mask)

            # Extend xgrid to low-x (N, x, 14) -> (N, 50, 14)
            xgrid_mask = np.zeros(large_xgrid.size, dtype=bool)
            offset = large_xgrid.size - xgrid.size
            for i in range(xgrid.size):
                xgrid_mask[offset + i] = True

            # This cast is needed if we want float64 fk tables. The reason being
            # that this mask is then applied to the padded fk tables, which are
            # float64. However, if we kept the mask float32, then the final padded
            # fk tables would be converted to float32 as well.
            # `tf.convert_to_tensor` became required after `compute_float_mask` adopted
            # `op.tensor_to_numpy_or_python` instead of `np.array`.
            xgrid_mask = tf.cast(
                compute_float_mask(tf.convert_to_tensor(xgrid_mask, dtype=tf.bool)),
                dtype=tf.float64,
            )
            paddedx_fk_table = op.einsum("Xx, nFx -> nXF", xgrid_mask, fk_table)
            xgrid_masks_dict[dataset_name] = xgrid_mask

            # Check the mask in x is applied correctly
            for i in range(large_xgrid.size):
                if i >= offset:
                    try:
                        assert np.allclose(
                            paddedx_fk_table[:, i, :], fk_table[:, :, i - offset]
                        )
                    except AssertionError:
                        print(f"Problem in the unchanged values for {dataset_name}")
                else:
                    try:
                        assert np.allclose(
                            paddedx_fk_table[:, i, :],
                            np.zeros_like(paddedx_fk_table[:, i, :]),
                        )
                    except AssertionError:
                        print(f"Problem in the extension for {dataset_name}")

            # Save to dict
            fk_table_dict[dataset_name] = paddedx_fk_table
            central_data_dict[dataset_name] = (
                exp_set.load_commondata().with_cuts(cuts).central_values.to_numpy()
            )

    res = result_tuple(
        fk_tables=fk_table_dict,
        central_data=central_data_dict,
        padded_fk=padded_fk_dict,
        xgrid_mask=xgrid_masks_dict,
    )
    return res
