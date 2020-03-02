from skimage import io, transform
import h5py
import dill
import numpy as np

import matplotlib.pyplot as plt

h5path = '/home/data_share/FBA/fall19/data/matrix/middle_2_3_matrix.h5'
id2idxpath = '/home/data_share/FBA/fall19/data/matrix/middle_id2idx.dill'
midi_op = 'aligned_s'

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

# Read scores from .dill files
mid_file = '/home/data_share/FBA/fall19/data/midi/{}_2_midi_{}_3.dill'.format(band, midi_op)
SC = dill.load(open(mid_file, 'rb')) # all scores / aligned midi

num = 3
sub_dim = 450

def createMatrixDill(PC, saveDill):
    data_dill = []
    ratings = []
    missing_aln = []
    for pc in PC:
        mtx = {}
        year = pc['year']
        student_id = pc['student_id']
        ratings = pc['ratings']  # choose which score
        mtx['ratings'] = ratings
        idx = id2idx[year][student_id]
        st, ed, dim = data['sc_idx'][idx]
        matrix = data['matrix'][int(st):int(ed), :int(dim)]

        try:
            aln = SC['alignment'][year][str(student_id)]
        except:
            print(year, student_id)
            missing_aln.append((year, student_id))
            continue

        # loop here
        matrix5 = []
        hop = aln.shape[0] / (num+1)
        window = aln.shape[0] / (num+1) * 2
        fac_midi = matrix.shape[0]/max(aln)
        for i in np.arange(num): # num sub_matrices of equal audio length, then resize to 300 x 300
            pc_st = int(i*hop/5)
            pc_ed = int((i*hop+window)/5)
            midi_st = int(aln[int(i*hop)]*fac_midi)
            midi_ed = int(aln[int(i*hop+window-1)]*fac_midi)
            #print(midi_st, midi_ed, matrix.shape)
            sub_matrix = matrix[midi_st:midi_ed, pc_st:pc_ed]
            sub_matrix = transform.resize(sub_matrix, (sub_dim, sub_dim))
            matrix5.append(sub_matrix)
            # if sum(ratings) > 2.4:
            #     plt.imshow(sub_matrix)
            #     plt.savefig(str(i)+'.png')
            #     print('saved!')

        mtx['matrix'] = np.array(matrix5)
        data_dill.append(mtx)

    with open(saveDill, 'wb') as f:
        dill.dump(data_dill, f)

    print(missing_aln)

createMatrixDill(trPC, '/home/data_share/FBA/fall19/data/matrix/matrix_chunk{}_train{}.dill'.format(num, sub_dim))
createMatrixDill(vaPC, '/home/data_share/FBA/fall19/data/matrix/matrix_chunk{}_valid{}.dill'.format(num, sub_dim))
createMatrixDill(tePC, '/home/data_share/FBA/fall19/data/matrix/matrix_chunk{}_test{}.dill'.format(num, sub_dim))

data.close()