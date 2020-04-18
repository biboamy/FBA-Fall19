from skimage import io, transform
import h5py
import dill, os
import numpy as np

# specify the band here
band = "symphonic"
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

if os.uname()[1] == 'mig1':
    PATH_FBA_DILL = "/media/SSD/FBA/saved_dill/"
    PATH_FBA_MIDI = "/media/SSD/FBA/fall19/data/midi/"
    PATH_FBA_MTX = "/media/SSD/FBA/fall19/data/matrix/"
    PATH_FBA_SPLIT = "/media/SSD/FBA/split_dill/"
    cpu_num = 3
else:
    PATH_FBA_DILL = "/media/Data/saved_dill/"
    PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
    PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
    PATH_FBA_SPLIT = "/media/Data/split_dill/"
    cpu_num = 5

h5path = '{}{}_{}_{}_matrix.h5'.format(PATH_FBA_MTX, band, SEGMENT, len(YEAR))
id2idxpath = '{}{}_id2idx_{}.dill'.format(PATH_FBA_MTX, band, len(YEAR))

id2idx = dill.load(open(id2idxpath, 'rb'))
data = h5py.File(h5path, 'r')

pc_file = '{}{}_2_pc_{}_'.format(PATH_FBA_SPLIT, band, len(YEAR))
# train
trPC = np.array(dill.load(open(pc_file + 'train.dill', 'rb')))
# valid
vaPC = np.array(dill.load(open(pc_file + 'valid.dill', 'rb')))
# test
tePC = np.array(dill.load(open(pc_file + 'test.dill', 'rb')))

def createMatrixDill(PC, target_dim, saveDill):
    data_dill = []
    ratings = []
    for pc in PC:
        mtx = {}
        year = pc['year']
        student_id = pc['student_id']
        ratings = pc['ratings']  # choose which score
        mtx['ratings'] = ratings

        if student_id not in id2idx[year].keys():
            print(year, student_id)
            continue

        idx = id2idx[year][student_id]
        st, ed, dim = data['sc_idx'][idx]
        matrix = data['matrix'][int(st):int(ed), :int(dim)].astype(np.float32)
        matrix = transform.resize(matrix, (target_dim, target_dim))
        mtx['matrix'] = matrix
        data_dill.append(mtx)

    with open(saveDill, 'wb') as f:
        dill.dump(data_dill, f)

for dim in [400, 600, 900]:
    createMatrixDill(trPC, dim, '{}{}_matrix_fixed_train{}.dill'.format(PATH_FBA_MTX, band, dim))
    createMatrixDill(vaPC, dim, '{}{}_matrix_fixed_valid{}.dill'.format(PATH_FBA_MTX, band, dim))
    createMatrixDill(tePC, dim, '{}{}_matrix_fixed_test{}.dill'.format(PATH_FBA_MTX, band, dim))

data.close()