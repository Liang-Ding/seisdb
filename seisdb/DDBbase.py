# -------------------------------------------------------------------
# The base class
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import os
from DSEM_Utils.DWidgets import get_proc_name
from DSEM_Utils.ibool_reader import DEquire_ibool

class DBbase():
    ''' The base class. '''

    def __init__(self, NSPEC, NGLL, model_dir):
        '''Initial parameters.'''
        self.NSPEC = NSPEC
        self.NGLL = NGLL
        self.model_dir = model_dir
        self._n_GLL_per_element = int(27)
        self._encoding_level = int(8)
        self.ibool_identifier = 'ibool.bin'

        # placeholder, mush be set in the child class.
        self.step0 = 0
        self.step1 = 0
        self.dstep = 0
        self.dNEZ_dirList = []
        self.NAME_DATA = 'data'


    def initial_paras(self, idx_processor, saving_dir, network, station):
        '''Initialize the parameters'''
        self.processor = get_proc_name(idx_processor=int(idx_processor))

        # create storing folder: ./path/network/station/
        self.saving_dir = saving_dir
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        self.saving_dir = os.path.join(self.saving_dir, network)
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        self.saving_dir = os.path.join(self.saving_dir, station)
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        # spatial sub-sampling
        self.ibool_file = os.path.join(self.model_dir, '%s_%s' % (self.processor, self.ibool_identifier))
        self.names_GLL_arr, self.index_GLL_arr = DEquire_ibool(self.ibool_file, self.NSPEC,
                                                               nGLL_per_element=self._n_GLL_per_element)

    def DCheck_valid_step(self):
        '''
        * Check and count the valid step.

        :param dir_array:       The directories containing the generated strain field.
                                MUST contain the 3 directories corresponding with the three unit forces.

        :param str_processor:   The name string of the Processor, eg:'proc0000*'.
        :param step0:           The starting step, not in second.       (int), eg: 0
        :param step1:           The ending step, not in second.         (int), eg: 6000
        :param dstep:           The interval of the time step, not in second.   (int), eg: 10
        :return:
                The array of indexes indicating the valid steps.
        '''

        self.valid_step_array = []
        for i_step in range(self.step0, self.step1, self.dstep):
            b_exist = True
            # check if the *.bin file exist.
            for data_dir in self.dNEZ_dirList:
                file = os.path.join(data_dir, '%s_%s_Step_%d.bin' %
                                    (self.processor, self.NAME_DATA, i_step))
                if os.path.exists(file) == False:
                    b_exist = False
            if b_exist:
                self.valid_step_array.append(i_step)

        # raise error if no data
        if len(self.valid_step_array) == 0:
            raise ValueError
