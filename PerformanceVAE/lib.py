import dill
import numpy as np
import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


def MSE_loss(pre, tar):
    loss_func = nn.MSELoss()
    pre = torch.sigmoid(pre)
    loss = loss_func(pre, tar)
    return loss


def compute_kld_loss(z_dist, prior_dist, beta, c=0.0):
    """

    :param z_dist: torch.distributions object
    :param prior_dist: torch.distributions
    :param beta: weight for kld loss
    :param c: capacity of bottleneck channel
    :return: kl divergence loss
    """
    kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
    kld = kld.sum(1).mean()
    kld = beta * (kld - c).abs()
    return kld


def load_data(band='middle', feat='pitch contour', midi_op='sec'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill
    assert(feat=='pitch contour')

    # Read features from .dill files
    pc_file = '/home/data_share/FBA/fall19/data/pitch_contour/{}_2_pc_3_'.format(band)
    # train
    trPC = np.array(dill.load(open(pc_file + 'train.dill', 'rb')))
    # valid
    vaPC = np.array(dill.load(open(pc_file + 'valid.dill', 'rb')))

    # Read scores from .dill files
    mid_file = '/home/data_share/FBA/fall19/data/midi/{}_2_midi_{}_3.dill'.format(band, midi_op)
    SC = dill.load(open(mid_file, 'rb')) # all scores / aligned midi

    return trPC, vaPC, SC


class Data2Torch(Dataset):
    def __init__(self, data, midi_op = 'aligned'):
        self.xPC = data[0]
        self.xSC = data[1]
        self.midi_op = midi_op
        self.resample = False
        if midi_op == 'resize':
            self.resample = True

        if midi_op == 'aligned' or midi_op == 'aligned_s':
            self.xSC = data[1]['scores']
            self.align = data[1]['alignment']

    def __getitem__(self, index):

        # pitch contour
        PC = self.xPC[index]['pitch_contour']
        mXPC = torch.from_numpy(PC).float()
        # ratings
        mY = torch.from_numpy(np.array([i for i in self.xPC[index]['ratings']])).float()
        mY = mY[0] # ratting order (0: musicality, 1: note accuracy, 2: rhythmetic, 3: tone quality)     

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
            oup = [mXPC, mXSC, mY]

        elif self.midi_op == 'aligned':
            year = self.xPC[index]['year']
            id = self.xPC[index]['student_id']
            mXSC = self.xSC[year][str(id)]
            mXSC = torch.from_numpy(mXSC).float()
            oup = [mXPC, mXSC, mY]

        elif self.midi_op == 'aligned_s':
            year = self.xPC[index]['year']
            instrument = self.xPC[index]['instrumemt']
            id = self.xPC[index]['student_id']
            SC =  self.xSC[instrument][year]
            SC = np.argmax(SC, axis=0)
            mXSC = torch.from_numpy(SC).float()
            align = self.align[year][str(id)]
            oup = [mXPC, mXSC, mY, align]

        else:
            raise ValueError('Please input the correct model')

        return oup
    
    def __len__(self):
        return len(self.xPC)

# padding each sequence in the batch to the same length
def my_collate(collate_params, batch):

    process_collate, sample_num, chunk_size = collate_params

    c_size = chunk_size

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
        num = sample_num

        for i, data in enumerate(batch):
            pc, sc = [], []
            for j in range(num):
                start = round(random.uniform(0, len(data[0])-c_size-10)) # -10: possible dismatch in size between pc & alignment
                pc.append(data[0][start:start+c_size].view(1,c_size))

                if len(data)>3:
                    idx = np.arange(np.floor(data[3][start]), np.floor(data[3][start+c_size]))
                    idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                    tmpsc = data[1][idx]
                    xval = np.linspace(0, idx.shape[0]-1, num=c_size)
                    x = np.arange(idx.shape[0])
                    sc_interp = np.interp(xval, x, tmpsc)

                    sc.append(torch.Tensor(sc_interp).view(1,-1))
                else:
                    sc.append(data[1][start:start+c_size].view(1,c_size))

                batch[i] = (torch.cat(pc,0), torch.cat(sc,0), data[2].repeat(num))

        return batch

    def window_chunk(batch):
        import random
        num = sample_num

        pc, sc, y = [], [], []
        for i, data in enumerate(batch):
            size = int((len(data[0])-10)/c_size) # -10: possible dismatch in size between pc & alignment
            for j in range(size):
                pc.append(data[0][j*c_size:j*c_size+c_size].view(1,c_size))
                if len(data)>3:
                    idx = np.arange(np.floor(data[3][j*c_size]), np.floor(data[3][j*c_size+c_size]))
                    idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                    tmpsc = data[1][idx]
                    xval = np.linspace(0, idx.shape[0]-1, num=c_size)
                    x = np.arange(idx.shape[0])
                    sc_interp = np.interp(xval, x, tmpsc)

                    sc.append(torch.Tensor(sc_interp).view(1,-1))
                else:
                    sc.append(data[1][j*c_size:j*c_size+c_size].view(1,j*c_size))
                y.append(data[2].view(1,1))
        c = list(zip(pc, sc, y))
        random.shuffle(c)
        pc, sc, y = zip(*c)
        pc = torch.cat(pc,0)
        sc = torch.cat(sc,0)
        y = torch.cat(y,0).squeeze()

        for i, data in enumerate(batch):
            batch[i] = (pc[i*num:i*num+num], sc[i*num:i*num+num], y[i*num:i*num+num])

        return batch

    if process_collate == 'padding':
        batch = window_chunk(batch)
    elif process_collate == 'randomChunk':
        batch = random_chunk(batch)
    else:
        assert (process_collate == 'windowChunk')
        batch = window_chunk(batch)
        
    return torch.utils.data.dataloader.default_collate(batch)