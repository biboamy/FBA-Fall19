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
    pc_file = '../../data_share/FBA/fall19/data/pitch_contour/{}_2_pc_3_'.format(band)
    # train
    trPC = np.array(dill.load(open(pc_file + 'train.dill', 'rb')))
    # valid
    vaPC = np.array(dill.load(open(pc_file + 'valid.dill', 'rb')))

    # Read scores from .dill files
    mid_file = '../../data_share/FBA/fall19/data/midi/{}_2_midi_3.dill'.format(band)
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
        # score feature, extract as a sequence
        SC =  np.argmax(self.xSC[instrument][year], axis=0)

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
        mY = mY[2] # ratting order (0: musicality, 1: note accuracy, 2: rhythmetic, 3: tone quality)

        return mXPC, mXSC, mY
    
    def __len__(self):
        return len(self.xPC)

# padding each sequence in the batch to the same length
def my_collate(batch):
    max_length_1 = 0
    max_length_2 = 0
    for data in batch:
        max_length_1 = max(data[0].shape[0],max_length_1)
        max_length_2 = max(data[1].shape[0],max_length_2)
    for i, data in enumerate(batch):
        data = (torch.nn.functional.pad(data[0], (0, max_length_1-data[0].shape[0]), "constant", 0), \
                torch.nn.functional.pad(data[1], (0, max_length_2-data[1].shape[0]), "constant", 0), \
                data[2])
        batch[i] = data
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# loss function, calculate the distane between two latent as the rating
def distance_loss(pitch_v, score_v, target):

    pdist = nn.PairwiseDistance(p=2)
    pred = pdist(pitch_v, score_v)

    loss_func = nn.MSELoss()
    loss = loss_func(pred, target)

    return loss

def classify_loss(pitch_v, score_v, target):

	return loss