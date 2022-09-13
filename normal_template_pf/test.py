from mrc_insar_common.data import data_reader
import glob
import tqdm
import numpy as np


def test(path, ext):
    bperp_paths = sorted(glob.glob('{}/*{}'.format(path, ext)))
    bperps = np.zeros(135)

    for idx in tqdm.tqdm(range(135)):
        # read delta days
        bperp_path = bperp_paths[idx]
        bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]
    return bperps


test("/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/bagdad.tsx.sm_dsc.1700.400.1500.1500/ifg_hr/", ".bperp")
