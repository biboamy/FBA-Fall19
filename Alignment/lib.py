import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset

def load_data(band='middle', feat='pitch contour'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill
    assert(feat=='pitch contour')

    # Read features from .dill files
    pc_file = './data/pitch_contour/{}_2_pc_3_'.format(band)
    # train
    trPC = np.array(dill.load(open(pc_file + 'train.dill', 'rb')))
    # valid
    vaPC = np.array(dill.load(open(pc_file + 'valid.dill', 'rb')))

    # Read scores from .dill files
    mid_file = './data/midi/{}_2_midi_3.dill'.format(band)
    SC = dill.load(open(mid_file, 'rb')) # all scores

    return trPC, vaPC, SC

class Data2Torch(Dataset):
    def __init__(self, data, resample=False):
        self.xPC = data[0]
        self.xSC = data[1]
        self.resample = resample

    def __getitem__(self, index):

        # pitch contour
        PC = self.xPC[index]['pitch_contour']
        mXPC = torch.from_numpy(PC).float()
        # musical score
        year = self.xPC[index]['year']
        instrument = self.xPC[index]['instrumemt']
        
        SC = self.xSC[instrument][year]

        # sample the midi to the length of audio
        if self.resample == True:
            l_target = PC.shape[1]
            t_midi = SC.get_end_time()
            mXSC = SC.get_piano_roll(fs = np.int(l_target / t_midi))
            l_midi = mXSC.shape[1]
            # pad 0 to ensure same length as feature
            mXSC = np.pad(mXSC, ((0,0),(0,l_target-l_midi)), 'constant')
            mXSC = torch.from_numpy(mXSC).float()
        else:
            mXSC = torch.from_numpy(SC).float()

        # ratings
        mY = torch.from_numpy(np.array([i for i in self.xPC[index]['ratings']])).float()

        return mXPC, mXSC, mY
    
    def __len__(self):
        return len(self.xPC)

def distance_loss(pitch_v, score_v, target):

	pdist = nn.PairwiseDistance(p=2)
	pred = pdist(pitch_v, score_v)

	loss_func = nn.MSELoss()
	loss = loss_func(pred, target)

	return loss

def classify_loss(pitch_v, score_v, target):

	return loss