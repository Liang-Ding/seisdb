# -------------------------------------------------------------------
# The class to Create the DGF database
# DGF: (Greens function, the displacement)
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import sys
import os
from seisdb.DDBbase import DBbase
from seisdb.DSEM_Utils.disp_reader import read_disp_bins

import numpy as np
import h5py
import zlib


class DDGFdb(DBbase):
    '''Class to create SGT database.'''

    def __init__(self, NSPEC, NGLL, model_dir):
        '''
        Initialize the model parameters!
        :param NSPEC:   The number of Spectral elements in the processor. (int), eg: 2232
        :param NGLL:    The number of Global GLL point each slice.
        :param model_dir:   The directory where the 3D model (*.bin files) is stored.
        '''

        super().__init__(NSPEC, NGLL, model_dir)
        self._n_dim = int(3)        # number of forces
        self._n_paras = int(3)      # number of element
        self.NAME_DATA = 'disp'     # The identifier set in SEM.
        self.dt = 0.1               # the time interval of the stored SGT database.
        self.data_name = 'dgf_data'
        self.data_header = 'dgf_header'


    # Function to create the DGF database.
    def create_db(self,dENZ_dirList, idx_processor,
                  step0, step1, dstep, saving_dir, network, station, dt):
        '''
        * Creating the DGF (displacement Greens function) database with sub-sampling and compression

        :param dNEZ_dirList:    The directories containing the strain field snapshot.
                                MUST contain the 3 directories associate with N,E,Z forces.
        :param idx_processor:   The index of the Processor, eg:'134'
        :param step0:           The starting time step, not in second.       (int), eg: 0
        :param step1:           The ending time step, not in second.         (int), eg: 6000
        :param dstep:           The interval of the time step, not in second.   (int), eg: 10

        :param saving_dir:      The output directory for saving the bin and info file.
        :param network:         The network of the station. (str) Eg: 'CI'
        :param station:         The station name. (str) Eg: 'USC'
        :param dt:              The time interval of the database in second. (float) Eg:  0.5

        :return:    True, if successfully save the database and info file.
        '''

        if len(dENZ_dirList) != 3:
            print("Require 3 folders in NEZ each storing the 3D wave field snapshot (the *.bin files)")
            raise ValueError

        # initialize
        self.initial_paras(idx_processor=idx_processor,
                        saving_dir=saving_dir,
                        network=network,
                        station=station)

        self.dNEZ_dirList = dENZ_dirList
        self.step0 = int(step0)
        self.step1 = int(step1)
        self.dstep = int(dstep)
        self.dt = dt

        # Check valid data snapshot
        self.DCheck_valid_step()

        # The # of selected GLL points each slice
        n_gll = len(self.names_GLL_arr)

        # count valid steps.
        n_step = len(self.valid_step_array)

        # allocate a buffer
        try:
            buffer = np.zeros((n_gll, n_step, self._n_dim, self._n_paras), dtype=np.float32)
        except:
            print("!!! The minimum RAM requirement is {} GB.".
                  format(n_gll * n_step * self._n_dim * self._n_paras * 4 / 1E9))
            print("!!! Unable to allocate sufficient memory!")
            raise ValueError

        # Extract the DGF at selected GLL points from the .bin files exported by the SEM.
        # loop for all step
        for idx, i_step in enumerate(self.valid_step_array):
            # data container for one step
            dat_arr_onestep = np.zeros((n_gll, self._n_paras, self._n_dim))

            # 3 unit forces at station (ENZ) --> the component of the syn. waveform.
            for i_dim, file_dir in enumerate(self.dNEZ_dirList):
                file_path = os.path.join(file_dir, '%s_%s_Step_%d.bin' %
                                         (self.processor, self.NAME_DATA, i_step))
                dat = read_disp_bins(file_path, self.NGLL, bin_type=np.float32)

                # 3C at gll point  --> the unit force for syn. waveform
                for i_para in range(self._n_paras):
                    dat_arr_onestep[:, i_para, i_dim] = dat[i_para, self.names_GLL_arr]

            dat_arr_onestep = np.array(dat_arr_onestep, dtype=np.float32)
            # use buffer on RAM to store data temporarily.
            buffer[:, idx, :, :] = dat_arr_onestep

        # Save the data and header.
        dgf_data_file = os.path.join(self.saving_dir, "%s_%s.bin" % (str(self.processor), self.data_name))
        header_file = os.path.join(self.saving_dir, "%s_%s.hdf5" % (str(self.processor), self.data_header))

        data_offset_array = []
        data_scale_array = []
        data_start_array = []
        data_length_array = []

        '''Compress and store the data (*.bin) and header (*.hdf5). '''
        with open(dgf_data_file, 'wb') as fw:
            for i in range(n_gll):
                data = np.empty(0)
                tmp_data = buffer[i]
                for j in range(self._n_paras):
                    for k in range(self._n_dim):
                        tmp = tmp_data[:, j, k]
                        # !!! ugly !!!, will be updated later.
                        data = np.hstack([data, tmp]).astype(np.float32)

                # 1. encoding
                # make all positive. [all amplitude >=0 ]
                offset_min = np.min(data)
                data = data - offset_min
                data_offset_array.append(offset_min)

                # Amplitude normalization => [0, 1]
                normal_factor = np.max(data)
                data = data / normal_factor
                data_scale_array.append(normal_factor)

                # encoding - convert the flota32 to uint16.
                if 8 == self._encoding_level:
                    data = np.asarray(data * (2 ** self._encoding_level - 1)).astype(np.uint8)
                else:
                    data = np.asarray(data * (2 ** self._encoding_level - 1)).astype(np.uint16)

                # 2. compress
                # numpy array to bytes
                data = np.ndarray.tobytes(data)
                # compress the byte data
                data_compress = zlib.compress(data)
                # the size of compressed DGF.
                size_compress_data = sys.getsizeof(data_compress)
                # the start position in byte.
                data_start_array.append(fw.tell())
                # the data length in byte.
                data_length_array.append(size_compress_data)
                # store the compressed DGF at selected global GLL points.
                fw.write(data_compress)

        # save the header into a separated HDF5 file.
        index_GLL_in_slice = self.names_GLL_arr
        data_start_array = np.asarray(data_start_array).astype(int)
        data_length_array = np.asarray(data_length_array).astype(int)
        data_offset_array = np.asarray(data_offset_array).astype(float)
        data_scale_array = np.asarray(data_scale_array).astype(float)

        with h5py.File(header_file, 'w') as f:
            f.create_dataset(name="index", shape=np.shape(index_GLL_in_slice), data=index_GLL_in_slice, dtype=int)
            f.create_dataset(name="start", shape=np.shape(data_start_array), data=data_start_array, dtype=int)
            f.create_dataset(name="length", shape=np.shape(data_length_array), data=data_length_array, dtype=int)
            f.create_dataset(name="offset", shape=np.shape(data_offset_array), data=data_offset_array, dtype=float)
            f.create_dataset(name="scale", shape=np.shape(data_scale_array), data=data_scale_array, dtype=float)

            f.attrs['ngll'] = n_gll
            f.attrs['nstep'] = n_step
            f.attrs['nforce'] = self._n_dim
            f.attrs['nparas'] = self._n_paras
            f.attrs['dt'] = self.dt
            f.attrs['nspec'] = self.NSPEC
            f.attrs['nGLL_global'] = self.NGLL
            f.attrs['type'] = 'DGF'
            f.attrs['forder'] = 'ENZ'
            f.attrs['version'] = '0.1.0'

        return True