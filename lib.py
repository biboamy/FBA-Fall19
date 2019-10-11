import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset

def load_data(band='middle', feat='pitch contour', midi_op='sec'):
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
    mid_file = '../../data_share/FBA/fall19/data/midi/{}_2_midi_{}_3.dill'.format(band, midi_op)
    SC = dill.load(open(mid_file, 'rb')) # all scores

    return trPC, vaPC, SC

def load_test_data(band='middle', feat='pitch contour'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill
    assert(feat=='pitch contour')

    # Read features from .dill files
    pc_file = '../../data_share/FBA/fall19/data/pitch_contour/{}_2_pc_3_'.format(band)
    # test
    tePC = np.array(dill.load(open(pc_file + 'test.dill', 'rb')))

    return tePC

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
        SC =  self.xSC[instrument][year]

        # sample the midi to the length of audio
        if self.resample == True:
            l_target = PC.shape[0]
            t_midi = SC.get_end_time()
            mXSC = SC.get_piano_roll(fs = np.int(l_target / t_midi))
            l_midi = mXSC.shape[1]
            # pad 0 to ensure same length as feature
            mXSC = np.pad(mXSC, ((0,0),(0,l_target-l_midi)), 'constant')
            mXSC = np.argmax(mXSC, axis=0)
            mXSC = torch.from_numpy(mXSC).float()
        else:
            SC = np.argmax(SC, axis=0)
            mXSC = torch.from_numpy(SC).float()

        # ratings
        mY = torch.from_numpy(np.array([i for i in self.xPC[index]['ratings']])).float()
        mY = mY[0] # ratting order (0: musicality, 1: note accuracy, 2: rhythmetic, 3: tone quality)
        
        return mXPC, mXSC, mY
    
    def __len__(self):
        return len(self.xPC)

# padding each sequence in the batch to the same length
def my_collate(batch):
 
    import random
    num = 3
  
    for i, data in enumerate(batch):
        pc = []
        sc = []
        for j in range(num):
            start = round(random.uniform(0, 1400))
            pc.append(data[0][start:start+1000].view(1,1000))
            sc.append(data[1][start:start+1000].view(1,1000))
  
        data = (torch.cat(pc,0), \
                torch.cat(sc,0), \
                data[2].repeat(num))
        
        batch[i] = data

    return torch.utils.data.dataloader.default_collate(batch)

# loss function, calculate the distane between two latent as the rating
def distance_loss(pitch_v, score_v, target):

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred = cos(pitch_v, score_v)

    loss_func = nn.MSELoss()
    loss = loss_func(pred, target.reshape(-1))
    return loss, pred

def get_weight(Ytr):
    mp = Ytr[:].sum(0).sum(1)
    mmp = mp.astype(np.float32) / mp.sum()
    cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
    cc[3]=1
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq