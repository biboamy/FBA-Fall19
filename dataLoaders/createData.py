import numpy as np
import pretty_midi
import pickle
import librosa

def midi_to_pianoroll(midifile):
    pm = pretty_midi.PrettyMIDI(midifile)
    pr = pm.get_piano_roll()

    name = midifile.split('/')[-1].split('.')[0]
    pkl_name = 'data/midi/' + name + 'pkl'

    # save piano roll to pickle
    f = open(pkl_name, 'wb')
    pickle.dump(pr, f)
    f.close()