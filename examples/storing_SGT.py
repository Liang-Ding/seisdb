# -------------------------------------------------------------------
# An example to create SGT database.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from seisdb.DSGT import DSGTdb

def create_SGT_database():
    '''
        Create the SGT database.
    '''

    dNEZ_dirList = ['the *.bin files generated by using the force in the direction of N',
                    'the *.bin files generated by using the force in the direction of E',
                    'the *.bin files generated by using the force in the direction of Z']

    NSPEC = int('Refer to the 3D model')
    NGLL = int('Refer to the 3D model')
    model_dir = str('The directory to the 3D model files (such as the *ibool.bin)')

    # the index of slice, staring from 0
    idx_processor = 120
    # the index of the first step
    step0 = 0
    # the index of the last step
    step1 = 18001
    # the step interval of the waveform field snapshot
    dstep = 50

    saving_dir = str('*/database/SGT/')
    # network name
    network = "CI"
    # station name
    station = 'DAN'
    dt = 0.5

    print("Working ... ")
    sgt = DSGTdb(NSPEC, NGLL, model_dir)
    sgt.create_db(dNEZ_dirList, idx_processor, step0, step1, dstep, saving_dir, network, station, dt)


if __name__ == '__main__':
    create_SGT_database()


