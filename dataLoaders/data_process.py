import os
import sys
from collections import defaultdict
import dill
import numpy as np
import scipy.io
import pretty_midi
from DataUtils import DataUtils

# Initialize input params, specify the band, intrument, segment information
BAND = ['middle', 'symphonic']
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015']

# define paths to FBA dataset and FBA annotations
# NEED TO EDIT THE PATH HERE IF USING ON A DIFFERENT COMPUTER
if sys.version_info[0] < 3:
    PATH_FBA_ANNO = '/Data/FBA2013/'
    PATH_FBA_AUDIO = '/Data/FBA2013data/'
    PATH_FBA_MIDI = '?'
else:
    PATH_FBA_ANNO = '/Users/caspia/Desktop/Github/FBA2013data/FBAAnnotations/'
    PATH_FBA_AUDIO = '/Users/caspia/Desktop/Github/FBA2013data/'
    PATH_FBA_MIDI = '/Users/caspia/Desktop/Github/FBA-Fall19/data/midi/'

# create midi data (piano roll)
midi_score = {}
for band in BAND:
    midi_score = {}
    for instrument in INSTRUMENT:
        midi_score[instrument] = {}
        for year in YEAR:
            midi_file_name = "{}_{}_{}.mid".format(year, band, instrument)
            pm = pretty_midi.PrettyMIDI(PATH_FBA_MIDI + midi_file_name)
            midi_score[instrument][year] = pm.get_piano_roll()

    print(len(midi_score))
    file_name = band + '_' + str(SEGMENT) + '_midi'
    if sys.version_info[0] < 3:
        with open('../data/midi/' + file_name + '.dill', 'wb') as f:
            dill.dump(midi_score, f)
        scipy.io.savemat('data/midi/' + file_name + '.mat', mdict={'perf_data': midi_score})
    else:
        with open('../data/midi/' + file_name + '_3.dill', 'wb') as f:
            dill.dump(midi_score, f)

# create data holder
perf_assessment_data = []
req_audio = True
# instantiate the data utils object for different instruments and create the data
for band in BAND:
    perf_assessment_data = []
    for instrument in INSTRUMENT:
        utils = DataUtils(PATH_FBA_ANNO, PATH_FBA_AUDIO, band, instrument)
        for year in YEAR:
            perf_assessment_data += utils.create_data(year, SEGMENT, audio=req_audio)
            print(len(perf_assessment_data))

    file_name = band + '_' + str(SEGMENT) + '_data'
    if sys.version_info[0] < 3:
        with open('../data/audio/' + file_name + '.dill', 'wb') as f:
            dill.dump(perf_assessment_data, f)
        scipy.io.savemat('data/audio/' + file_name + '.mat', mdict={'perf_data': perf_assessment_data})
    else:
        with open('../data/audio/' + file_name + '_3.dill', 'wb') as f:
            dill.dump(perf_assessment_data, f)