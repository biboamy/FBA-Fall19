import os
import sys
from collections import defaultdict
import dill
import numpy as np
import scipy.io
import pypianoroll
import pretty_midi
from DataUtils import DataUtils
import h5py
from scipy.signal import decimate
import time
from tqdm import tqdm
from lib import load_data, load_test_data

# Initialize input params, specify the band, intrument, segment information
BAND = ['middle']
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015']

feat = 'pitch contour'
midi_op = 'res12'

if sys.version_info[0] < 3:
    PATH_FBA_ANNO = '/Data/FBA2013/'
    PATH_FBA_AUDIO = '/Data/FBA2013data/'
    PATH_FBA_MIDI = '../data/midi/'
else:
    PATH_FBA_ANNO = '/Users/caspia/Desktop/Github/FBA2013data/FBAAnnotations/'
    PATH_FBA_AUDIO = '/Users/caspia/Desktop/Github/FBA2013data/'
    PATH_FBA_MIDI = '/Users/caspia/Desktop/Github/FBA-Fall19/data/midi/'

def computeDistanceMatrix(perf):

    year = perf['year']
    instrument = perf['instrumemt']
    pc = perf['pitch_contour']
    sc = SC[instrument][year]
    sc = np.argmax(sc, axis=0)
    silence_sc = (sc==0)

    pc = decimate(pc, 5) # downsample factor: 5

    silence_pc = (pc < 1)
    pc[pc < 1] = 1
    wav_pitch_contour_in_midi = 69 + 12 * np.log2(pc / 440);
    wav_pitch_contour_in_midi[silence_pc] = 0
    pc = wav_pitch_contour_in_midi

    N = 12 # octave
    D = np.zeros((len(sc), len(pc)))
    for i in np.arange(len(sc)):
        for j in np.arange(len(pc)):
            diff = np.abs(sc[i] - pc[j])
            D[i, j] = diff % N
            D[i, j] = np.min([D[i, j], 12 - D[i, j]]) + np.min([1, np.floor(diff / N)])

    # value for (silence, non-silence) pair
    p = np.mean(D[~silence_sc,:][:, ~silence_pc])
    # replace the values in the matrix
    D[np.ix_(silence_sc, ~silence_pc)] = p
    D[np.ix_(~silence_sc, silence_pc)] = p

    D = D.astype(np.float16)
    perf['matrix'] = D

    return perf


# create data holder
matrix_data = []
from joblib import Parallel, delayed
# instantiate the data utils object for different instruments and create the data
for band in BAND:
    break
    trPC, vaPC, SC = load_data(band, feat, midi_op)
    tePC = load_test_data(band, feat)

    PC = np.concatenate((trPC,vaPC,tePC),axis=0)

    p = np.array([0, 282, 564, 846, 1128, 1410])
    for i in np.arange(0,5):
        print(i)
        del matrix_data
        start = time.time()
        matrix_data = Parallel(n_jobs=5)(delayed(computeDistanceMatrix)(perf) for perf in tqdm(PC[p[i]:p[i+1]]))
        stop = time.time()
        print('Elapsed time for the entire processing: {:.2f} s'
              .format(stop - start))
        print('{} to {}'.format(p[i], p[i+1]))

        file_name = band + '_' + str(SEGMENT) + '_matrix_' + str(i)
        with open('../../../data_share/FBA/fall19/data/matrix/' + file_name + '.dill', 'wb') as f:
            dill.dump(matrix_data, f)


# save matrix to an h5py file
sc_dim = 500
pc_dim = 6931
cnt = 0

f = h5py.File('../../../data_share/FBA/fall19/data/matrix/{}_{}_3_matrix.h5'.format(band, SEGMENT), 'w')
f.create_dataset('matrix', (0, pc_dim), maxshape=(None, pc_dim))
f.create_dataset('sc_idx', (0, 3), maxshape=(None, 3)) # (i, j, k) -> f['matrix'][i:j, 0:k]
id2idx = {'2013':{}, '2014':{}, '2015':{}}

ds_mtx = f['matrix']
ds_scidx = f['sc_idx']

def write_incre_h5(dataset, datapoint):
    dataset.resize(dataset.shape[0]+datapoint.shape[0],axis=0)
    dataset[-datapoint.shape[0]:,:] = datapoint

for i in np.arange(5):
    file_name = band + '_' + str(SEGMENT) + '_matrix_' + str(i)
    tmpdillfile = '../../../data_share/FBA/fall19/data/matrix/' + file_name + '.dill'
    performance = np.array(dill.load(open(tmpdillfile, 'rb')))
    for perf in performance:
        # get max dim
        mtx_cur = perf['matrix']

        id2idx[perf['year']][perf['student_id']] = cnt

        sc_dim_i, pc_dim_i = mtx_cur.shape
        #if sc_dim_i > sc_dim:
        #    sc_dim = sc_dim_i
        #if pc_dim_i > pc_dim:
        #    pc_dim = pc_dim_i

        i, j, k = ds_mtx.shape[0], ds_mtx.shape[0]+sc_dim_i, pc_dim_i
        write_incre_h5(ds_scidx, np.array([i, j, k]))
        # padding
        mtx_cur = np.pad(mtx_cur, ((0,0),(0,pc_dim-pc_dim_i)), 'constant', constant_values=0)
        assert mtx_cur.shape[1] == pc_dim
        write_incre_h5(ds_mtx, mtx_cur)
        assert ds_mtx.shape[0] == j

        cnt = cnt + 1

print(sc_dim, pc_dim)
print("write {} matrices.".format(cnt))
f.close()

print(id2idx)
with open('../../../data_share/FBA/fall19/data/matrix/' + band + '_id2idx.dill', 'wb') as f:
    dill.dump(id2idx, f)

