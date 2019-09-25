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
perf_assessment_midi = []
for band in BAND:
    perf_assessment_midi = []
    for instrument in INSTRUMENT:
        for year in YEAR:
            midi_score = {}
            midi_score['year'] = year
            midi_score['band'] = band
            midi_score['instrument'] = instrument
            midi_score['segment'] = SEGMENT
            midi_file_name = "{}_{}_{}.midi".format(year, band, instrument)
            pm = pretty_midi.PrettyMIDI(PATH_FBA_MIDI + midi_file_name)
            midi_score['piano_roll'] = pm.get_piano_roll()
            perf_assessment_midi.append(midi_score)

    print(len(perf_assessment_midi))
    file_name = band + '_' + str(SEGMENT) + '_midi'
    if sys.version_info[0] < 3:
        with open('../data/audio/' + file_name + '.dill', 'wb') as f:
            dill.dump(perf_assessment_midi, f)
        scipy.io.savemat('data/audio/' + file_name + '.mat', mdict={'perf_data': perf_assessment_midi})
    else:
        with open('../data/audio/' + file_name + '_3.dill', 'wb') as f:
            dill.dump(perf_assessment_midi, f)

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