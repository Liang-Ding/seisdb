# -------------------------------------------------------------------
# Functions to read the *disp*.bin files.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from DSEM_Utils.bin_reader import read_bin_by_scipy
import numpy as np


def read_disp_bins(file_path, nGLL_global, bin_type=np.float32):
    '''
    * Return the displacement field export by the Specfem3D dl_runtime_saver.f90

    :param file_path:       The file path of the bin file.
    :param nGLL_global:     The total number of GLL points each slice.
    :param bin_type:        The data type, usually is single-precision float(32 bit).
    :return:                The epsilon field (Exx, Eyy, Ezz, Exy, Exz, Eyz)
    '''

    n_channel = 3
    dat = read_bin_by_scipy(file_path, file_type=bin_type)
    return np.reshape(dat, (nGLL_global, n_channel)).transpose()


