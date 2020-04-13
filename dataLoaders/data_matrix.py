import os
import sys
from collections import defaultdict
import dill
import numpy as np
import scipy.io
import pypianoroll
import pretty_midi
import h5py
from scipy.signal import decimate
import time
from tqdm import tqdm

# Initialize input params, specify the band, intrument, segment information
BAND = ['symphonic']
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

dill_name = {'middle': 'middle_2_new_dataPC.dill', 'symphonic': 'symphonic_2_data_new2.dill'}
total_num = {'middle': np.array([0, 523, 1046, 1569, 2092, 2616]), 'symphonic': np.array([0, 304, 608, 912, 1216, 1520, 1824, 2128, 2432, 2736, 3040])}

feat = 'pitch contour'
midi_op = 'res12'

fail_list = []

if os.uname()[1] == 'mig1':
    PATH_FBA_DILL = "/media/SSD/FBA/saved_dill/"
    PATH_FBA_MIDI = "/media/SSD/FBA/fall19/data/midi/"
    PATH_FBA_MTX = "/media/SSD/FBA/fall19/data/matrix/"
    cpu_num = 3
else:
    PATH_FBA_DILL = "/media/Data/saved_dill/"
    PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
    PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
    cpu_num = 5

def load_data(band='middle', midi_op='res12'):
    import dill
    assert (feat == 'pitch contour')

    # Read features from .dill files
    pc_file = PATH_FBA_DILL + dill_name[band]
    PC = np.array(dill.load(open(pc_file, 'rb')))

    # Read scores from .dill files
    mid_file = PATH_FBA_MIDI + '{}_2_midi_{}_6.dill'.format(band, midi_op)
    SC = dill.load(open(mid_file, 'rb'))  # all scores / aligned midi

    return PC, SC

def simpleDTW(D):
    m, n = D.shape
    dir_vec = [[1,0],[0,1],[1,1]]
    dir = np.zeros_like(D)
    cum = np.zeros_like(D)
    path = [[m-1, n-1]]

    cum[0,0] = D[0,0]
    for i in range(1,m):
        dir[i, 0] = 0
        cum[i, 0] = cum[i-1,0]+D[i,0]

    for j in range(n):
        dir[0, j] = 1
        cum[0, j] = cum[0,j-1]+D[0,j]

    for i in np.arange(1,m):
        for j in np.arange(1,n):
            dir[i,j] = np.argmin([cum[i-1,j], cum[i,j-1], cum[i-1,j-1]])
            cum[i,j] = np.min([cum[i-1,j], cum[i,j-1], cum[i-1,j-1]]) + D[i,j]

    # retrieve path
    x, y = m-1, n-1
    while(x > 0 or y > 0):
        inc_dir = int(dir[x, y])
        x = x - dir_vec[inc_dir][0]
        y = y - dir_vec[inc_dir][1]
        path.append([x, y])

    path.reverse()
    return np.array(path)


def computeDistanceMatrixAndAlignment(perf):

    #print(perf['year'], perf['student_id'])
    try:
        year = perf['year']
        instrument = perf['instrumemt']
        pc = perf['pitch_contour']
        # remove audio
        perf['audio'] = []
        sc = SC[instrument][year]
        sc = np.argmax(sc, axis=0)
        silence_sc = (sc==0)

        pc = decimate(pc, 5) # downsample factor: 5

        idx_sc = np.array([i for i in range(sc.shape[0]) if sc[i] != 0])
        idx_pc = np.array([i for i in range(pc.shape[0]) if pc[i] > 1])

        silence_pc = (pc < 1)
        pc[pc < 1] = 1

        wav_pitch_contour_in_midi = 69 + 12 * np.log2(pc / 440);
        wav_pitch_contour_in_midi[silence_pc] = 0

        N = 12 # octave
        D = np.zeros((len(sc), len(wav_pitch_contour_in_midi)))
        for i in np.arange(len(sc)):
            for j in np.arange(len(wav_pitch_contour_in_midi)):
                diff = np.abs(sc[i] - wav_pitch_contour_in_midi[j])
                D[i, j] = diff % N
                D[i, j] = np.min([D[i, j], 12 - D[i, j]]) + np.min([1, np.floor(diff / N)])

        # dtw
        path = simpleDTW(D[~silence_sc,:][:, ~silence_pc])

        pc_to_sc = np.zeros_like(idx_pc)
        for i in np.arange(len(path)-1, -1, -1):
            pc_to_sc[path[i,1]] = idx_sc[path[i,0]]

        alignment = np.zeros_like(pc) - 1
        alignment[idx_pc] = pc_to_sc

        st = 0
        ed = st+1
        while(st < pc.shape[0]):
            # go through non-silence alignments
            while (st < pc.shape[0] and alignment[st] != -1):
                st = st + 1
            if st >= pc.shape[0]:
                break
            ed = st + 1
            while (ed < pc.shape[0] and alignment[ed] == -1):
                ed = ed + 1
            lin_st = 0
            lin_ed = 0
            if ed == pc.shape[0]:
                lin_ed = sc.shape[0]-1
            else:
                lin_ed = alignment[ed]
            if st == 0:
                lin_st = 0
            else:
                lin_st = alignment[st-1]
            lin_aln = np.linspace(lin_st, lin_ed, ed-st+2)
            alignment[st:ed] = lin_aln[1:-1]

        assert np.sum(np.diff(alignment)>=0) == alignment.shape[0]-1

        # value for (silence, non-silence) pair
        p = np.mean(D[~silence_sc,:][:, ~silence_pc])
        # replace the values in the matrix
        D[np.ix_(silence_sc, ~silence_pc)] = p
        D[np.ix_(~silence_sc, silence_pc)] = p
        D[np.ix_(silence_sc, silence_pc)] = p

        D = D.astype(np.float16)
        perf['matrix'] = D
        #print(D.shape)
        perf['alignment'] = alignment
    except:
        fail_list.append(perf)
    return perf


# create data holder
matrix_data = []
from joblib import Parallel, delayed
# instantiate the data utils object for different instruments and create the data
for band in BAND:
    PC, SC = load_data(band=band, midi_op=midi_op)
    print(len(PC))
    p = total_num[band]
    for i in np.arange(len(p)-1):
        print(i)
        del matrix_data
        start = time.time()
        matrix_data = Parallel(n_jobs=cpu_num)(delayed(computeDistanceMatrixAndAlignment)(perf) for perf in tqdm(PC[p[i]:p[i+1]]))
        stop = time.time()
        print('Elapsed time for the entire processing: {:.2f} s'
              .format(stop - start))
        print('{} to {}'.format(p[i], p[i+1]))

        file_name = band + '_' + str(SEGMENT) + '_matrix_r_' + str(len(YEAR)) + "_" + str(i)
        with open(PATH_FBA_MTX + file_name + '.dill', 'wb') as f:
            dill.dump(matrix_data, f)

    with open(PATH_FBA_MTX + '{}_fail_list.dill'.format(band), 'wb') as f:
        dill.dump(fail_list, f)

# band = 'middle'

# save matrix to an h5py file
# sc_dim = 500
# pc_dim = 6931
# cnt = 0
#
# f = h5py.File('../data/matrix/{}_{}_6_matrix.h5'.format(band, SEGMENT), 'w')
# f.create_dataset('matrix', (0, pc_dim), maxshape=(None, pc_dim))
# f.create_dataset('sc_idx', (0, 3), maxshape=(None, 3)) # (i, j, k) -> f['matrix'][i:j, 0:k]
# id2idx = {'2013':{}, '2014':{}, '2015':{}, '2016':{}, '2017':{}, '2018':{}}
#
# ds_mtx = f['matrix']
# ds_scidx = f['sc_idx']
#
# def write_incre_h5(dataset, datapoint):
#     dataset.resize(dataset.shape[0]+datapoint.shape[0],axis=0)
#     dataset[-datapoint.shape[0]:,:] = datapoint
#
# for i in np.arange(len(total_num[band])-1):
#     file_name = band + '_' + str(SEGMENT) + '_matrix_3_' + str(i)
#     tmpdillfile = '../data/matrix/' + file_name + '.dill'
#     print(tmpdillfile)
#     performance = np.array(dill.load(open(tmpdillfile, 'rb')))
#     for perf in performance:
#         if 'matrix' not in perf.keys():
#             print(perf['year'], perf['student_id'])
#             continue
#         # get max dim
#         mtx_cur = perf['matrix']
#
#         id2idx[perf['year']][perf['student_id']] = cnt
#
#         sc_dim_i, pc_dim_i = mtx_cur.shape
#         if sc_dim_i > sc_dim:
#             sc_dim = sc_dim_i
#         if pc_dim_i > pc_dim:
#             pc_dim = pc_dim_i
#         continue
#
#         i, j, k = ds_mtx.shape[0], ds_mtx.shape[0]+sc_dim_i, pc_dim_i
#         write_incre_h5(ds_scidx, np.array([[i, j, k]]))
#         # padding
#         mtx_cur = np.pad(mtx_cur, ((0,0),(0,pc_dim-pc_dim_i)), 'constant', constant_values=0)
#         assert mtx_cur.shape[1] == pc_dim
#         write_incre_h5(ds_mtx, mtx_cur)
#         assert ds_mtx.shape[0] == j
#
#         cnt = cnt + 1
#
# print(sc_dim, pc_dim)
# print("write {} matrices.".format(cnt))
# f.close()
#
# print(id2idx)
# with open('/home/data_share/FBA/fall19/data/matrix/' + band + '_id2idx.dill', 'wb') as f:
#     dill.dump(id2idx, f)

