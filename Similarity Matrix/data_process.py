from skimage import io, transform
import h5py
import dill
import numpy as np

h5path = '../../../data_share/FBA/fall19/data/matrix/middle_2_3_matrix.h5'
id2idxpath = '../../../data_share/FBA/fall19/data/matrix/middle_id2idx.dill'

id2idx = dill.load(open(id2idxpath, 'rb'))
data = h5py.File(h5path, 'r')

band = 'middle'
pc_file = '../../../data_share/FBA/fall19/data/pitch_contour/{}_2_pc_3_'.format(band)
# train
trPC = np.array(dill.load(open(pc_file + 'train.dill', 'rb')))
# valid
vaPC = np.array(dill.load(open(pc_file + 'valid.dill', 'rb')))
# test
tePC = np.array(dill.load(open(pc_file + 'test.dill', 'rb')))

def createMatrixDill(PC, saveDill):
    data_dill = []
    ratings = []
    for pc in PC:
        mtx = {}
        year = pc['year']
        student_id = pc['student_id']
        ratings = pc['ratings']  # choose which score
        mtx['ratings'] = ratings
        idx = id2idx[year][student_id]
        st, ed, dim = data['sc_idx'][idx]
        matrix = data['matrix'][int(st):int(ed), :int(dim)]
        matrix = transform.resize(matrix, (600, 600))
        mtx['matrix'] = matrix
        data_dill.append(mtx)

    with open(saveDill, 'wb') as f:
        dill.dump(data_dill, f)

createMatrixDill(trPC, '/home/data_share/FBA/fall19/data/matrix/matrix_fixed_train.dill')
createMatrixDill(vaPC, '/home/data_share/FBA/fall19/data/matrix/matrix_fixed_valid.dill')
createMatrixDill(tePC, '/home/data_share/FBA/fall19/data/matrix/matrix_fixed_test.dill')


data.close()