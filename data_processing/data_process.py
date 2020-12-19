import dill
import scipy.io
import pypianoroll
import pretty_midi
from DataUtils import DataUtils
import scipy.io

# Initialize input params, specify the band, intrument, segment information
BAND = ['middle', 'symphonic']
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

# define paths to FBA dataset and FBA annotations
# NEED TO EDIT THE PATH HERE IF USING ON A DIFFERENT COMPUTER
PATH_FBA_ANNO = '/media/SSD/FBA/MIG-FbaData/'
PATH_FBA_AUDIO = '' # not including raw audio
PATH_FBA_MIDI = "/media/SSD/FBA/fall19/data/midi/"
PATH_FBA_DILL = "/media/SSD/FBA/save_dill/"

# create data holder
perf_assessment_data = []
req_audio = False
req_rating = True
# instantiate the data utils object for different instruments and create the data
for band in BAND:
    perf_assessment_data = []
    for instrument in INSTRUMENT:
        utils = DataUtils(PATH_FBA_ANNO, PATH_FBA_AUDIO, band, instrument)
        for year in YEAR:
            perf_assessment_data += utils.create_data(year, SEGMENT, audio=req_audio, rating=req_rating)

    file_name = band + '_' + str(SEGMENT) + '_pc_' + str(len(YEAR))
    print("Saving to " + file_name)
    with open(PATH_FBA_DILL + file_name + '.dill', 'wb') as f:
        dill.dump(perf_assessment_data, f)

# create midi data (piano roll)
midi_score = {}
unit = "res12"
for band in BAND:
    midi_score = {}
    for instrument in INSTRUMENT:
        midi_score[instrument] = {}
        for year in YEAR:
            midi_file_name = "{}_{}_{}.mid".format(year, band, instrument)
            if unit == "sec":
                pm = pretty_midi.PrettyMIDI(PATH_FBA_MIDI + midi_file_name)
                midi_score[instrument][year] = pm.get_piano_roll()
            elif unit == "beat":
                pm = pypianoroll.Multitrack(PATH_FBA_MIDI + midi_file_name)
                midi_score[instrument][year] = pm.tracks[0].pianoroll.T
            elif unit == "res12":
                pm = pypianoroll.Multitrack(PATH_FBA_MIDI + midi_file_name, beat_resolution=12)
                midi_score[instrument][year] = pm.tracks[0].pianoroll.T
            elif unit == "resize":
                pm = pretty_midi.PrettyMIDI(PATH_FBA_MIDI + midi_file_name)
                midi_score[instrument][year] = pm
            print(instrument, year, midi_score[instrument][year].shape)
    print("Loaded {} midi files.".format(len(midi_score)))
    file_name = band + '_' + str(SEGMENT) + '_midi_' + unit + "_" + str(len(YEAR))

    print("Saving to " + file_name)
    with open(PATH_FBA_MIDI + file_name + '.dill', 'wb') as f:
        dill.dump(midi_score, f)
