import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

def check_missing_alignedmidi(band='middle', feat='pitch contour', midi_op='sec'):
    # find missing alignedmidi

    import dill
    pc_file = '../../data_share/FBA/fall19/data/pitch_contour/{}_2_pc_3.dill'.format(band)
    all_PC = np.array(dill.load(open(pc_file, 'rb')))

    mid_file = '../../data_share/FBA/fall19/data/midi/{}_2_midi_{}_3.dill'.format(band, midi_op)
    SC = dill.load(open(mid_file, 'rb'))  # all scores / aligned midi

    missing_list = []
    for i in np.arange(all_PC.shape[0]):
        year = all_PC[i]['year']
        id_PC = all_PC[i]['student_id']
        if str(id_PC) not in SC[year].keys():
            missing_list.append('{}.{}'.format(year, id_PC))

    print(missing_list)

    return missing_list


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
    SC = dill.load(open(mid_file, 'rb')) # all scores / aligned midi

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
    def __init__(self, data, midi_op = 'aligned'):
        self.xPC = data[0]
        self.xSC = data[1]
        self.align = {}
        self.midi_op = midi_op
        self.resample = False
        if midi_op == 'resize':
            self.resample = True

        if midi_op == 'aligned' or midi_op == 'aligned_s':
            self.xSC = data[1]['scores']
            self.align = data[1]['alignment']

    def __getitem__(self, index):

        align = [] # empty except midi_op = 'aligned'

        # pitch contour
        PC = self.xPC[index]['pitch_contour']
        mXPC = torch.from_numpy(PC).float()
        align = None

        if self.midi_op in ['sec', 'beat', 'resize']:
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

        elif self.midi_op == 'aligned':
            year = self.xPC[index]['year']
            id = self.xPC[index]['student_id']
            mXSC = self.xSC[year][str(id)]
            mXSC = torch.from_numpy(mXSC).float()

        elif self.midi_op == 'aligned_s':
            year = self.xPC[index]['year']
            instrument = self.xPC[index]['instrumemt']
            id = self.xPC[index]['student_id']
            SC =  self.xSC[instrument][year]
            SC = np.argmax(SC, axis=0)
            mXSC = torch.from_numpy(SC).float()

            align = self.align[year][str(id)]

        else:
            raise ValueError('Please input the correct model')

        # ratings
        mY = torch.from_numpy(np.array([i for i in self.xPC[index]['ratings']])).float()
        mY = mY[0] # ratting order (0: musicality, 1: note accuracy, 2: rhythmetic, 3: tone quality)
        
        return mXPC, mXSC, mY, align

    def __len__(self):
        return len(self.xPC)

# padding each sequence in the batch to the same length
def my_collate(batch):

    def padding(batch):
        max_length = 0
        for i, data in enumerate(batch):
            max_tmp = max(len(data[0]),len(data[1]))
            max_length = max(max_length, max_tmp)
        for i, data in enumerate(batch):
            PC, SC = data[0], data[1]
            PC = F.pad(PC, (0, max_length-len(PC)), "constant", 0)
            SC = F.pad(SC, (0, max_length-len(SC)), "constant", 0)
            batch[i] = (PC, SC, data[2])
        return batch        

    def random_chunk(batch):
        import random
        num = 3

        for i, data in enumerate(batch):
            pc, sc = [], []
            for j in range(num):
                start = round(random.uniform(0, 1400))
                pc.append(data[0][start:start+1000].view(1,1000))

                if data[3]:
                    idx = np.arange(np.floor(data[3][start]), np.floor(data[3][start+1000]))
                    idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                    tmpsc = data[1][idx]
                    xval = np.linspace(0, idx.shape[0]-1, num=1000)
                    x = np.arange(idx.shape[0])
                    sc_interp = np.interp(xval, x, tmpsc)

                    sc.append(torch.Tensor(sc_interp).view(1,-1))
                else:
                    sc.append(data[1][start:start+1000].view(1,1000))

            batch[i] = (torch.cat(pc,0), \
                        torch.cat(sc,0), \
                        data[2].repeat(num))
        return batch

    def window_chunk(batch):
        import random
        num = 3

        pc, sc, y = [], [], []
        for i, data in enumerate(batch):
            size = int(len(data[0])/2000)
            for j in range(size):
                pc.append(data[0][j*2000:j*2000+2000].view(1,2000))
                sc.append(data[1][j*2000:j*2000+2000].view(1,2000))
                y.append(data[2].view(1,1))

        c = list(zip(pc, sc, y))
        random.shuffle(c)
        pc, sc, y = zip(*c)
        pc = torch.cat(pc,0)
        sc = torch.cat(sc,0)
        y = torch.cat(y,0).squeeze()
  
        for i, data in enumerate(batch):
            batch[i] = (pc[i*3:i*3+3], \
                        sc[i*3:i*3+3], \
                        y[i*3:i*3+3])
        return batch
    
    #batch = padding(batch)
    batch = random_chunk(batch)
    #batch = window_chunk(batch)
        
    return torch.utils.data.dataloader.default_collate(batch)

def test_collate(batch):

    def non_overlap(batch):
        pc, sc, y = [], [], []
        for i, data in enumerate(batch):
            size = int(len(data[0])/1000)
            for j in range(size):
                pc.append(data[0][j*1000:j*1000+1000].view(1,1000))
                sc.append(data[1][j*1000:j*1000+1000].view(1,1000))
                y.append(data[2].view(1,1))
            pc.append(data[0][-1000:].view(1,1000))
            sc.append(data[1][-1000:].view(1,1000))
            y.append(data[2].view(1,1))
            pc = torch.cat(pc,0)
            sc = torch.cat(sc,0)
            y = torch.cat(y,0).squeeze()
            batch[i] = (pc, sc, y)
        return batch

    def overlap(batch, hopSize):
        pc, sc, y = [], [], []
        for i, data in enumerate(batch):
            j = 0
            while (j+1000) < len(data[0]):
                pc.append(data[0][j:j+1000].view(1,1000))
                sc.append(data[1][j:j+1000].view(1,1000))
                y.append(data[2].view(1,1))
                j+=hopSize
            pc.append(data[0][-1000:].view(1,1000))
            sc.append(data[1][-1000:].view(1,1000))
            y.append(data[2].view(1,1))
            pc = torch.cat(pc,0)
            sc = torch.cat(sc,0)
            y = torch.cat(y,0).squeeze()
            batch[i] = (pc, sc, y)
        return batch

    batch=overlap(batch, 500)
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