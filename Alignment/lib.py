import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from config import *

def check_missing_alignedmidi(band='middle', feat='pitch contour', midi_op='res12'):
    # find missing alignedmidi

    import dill
    pc_file = PATH_FBA_DILL + data_all_pc.format(band)
    all_PC = np.array(dill.load(open(pc_file, 'rb')))

    mid_file = PATH_FBA_MIDI + midi_aligned_s.format(band)
    SC = dill.load(open(mid_file, 'rb'))  # all scores / aligned midi

    missing_list = []
    for i in np.arange(all_PC.shape[0]):
        inst = all_PC[i]['instrumemt']
        year = all_PC[i]['year']
        id_PC = all_PC[i]['student_id']
        if id_PC not in SC['alignment'][year].keys():
            missing_list.append((year, id_PC))

    print(missing_list)

    return missing_list

def normalize_pc_and_sc(pc, sc):
    silence_pc = (pc < 1)
    pc[pc < 1] = 1

    ret_pc = 69 + 12 * np.log2(pc / 440);
    ret_pc[silence_pc] = 0

    ret_pc = ret_pc / 128
    sc = sc / 128

    return ret_pc, sc

def load_data(band='middle', feat='pitch contour', midi_op='res12'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill
    assert(feat=='pitch contour')

    # train
    trPC = np.array(dill.load(open(PATH_FBA_SPLIT + data_train_pc[split].format(band), 'rb')))
    # valid
    vaPC = np.array(dill.load(open(PATH_FBA_SPLIT + data_valid_pc[split].format(band), 'rb')))

    # Read scores from .dill files
    mid_file = PATH_FBA_MIDI + midi_aligned_s.format(band)
    SC = dill.load(open(mid_file, 'rb')) # all scores / aligned midi

    return trPC, vaPC, SC

def load_test_data(band='middle', feat='pitch contour'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill
    assert(feat=='pitch contour')

    # test
    tePC = np.array(dill.load(open(PATH_FBA_SPLIT + data_test_pc[split].format(band), 'rb')))

    return tePC

class Data2Torch(Dataset):
    def __init__(self, data, midi_op = 'aligned_s'):
        self.xPC = data[0]
        self.xSC = data[1]
        self.midi_op = midi_op
        self.resample = False

        assert midi_op == 'aligned_s'

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
        mY = mY[score_choose] # ratting order (0: musicality, 1: note accuracy, 2: rhythmetic, 3: tone quality)

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
            align = self.align[year][id]
            if normalize:
                mXPC, mXSC = normalize_pc_and_sc(mXPC, mXSC)
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
                    idx = np.arange(np.floor(data[3][np.int(start/5)]), np.floor(data[3][np.int((start+c_size)/5)]+1))
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
                    idx = np.arange(np.floor(data[3][np.int(j*c_size/5)]), np.floor(data[3][np.int((j*c_size+c_size)/5)]+1))
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

def test_collate(collate_params, batch):

    overlap_flag, chunk_size = collate_params

    c_size = chunk_size

    def non_overlap(batch):
        pc, sc, y = [], [], []
        for i, data in enumerate(batch):
            size = int((len(data[0])-10)/c_size) # -10: possible dismatch in size between pc & alignment
            for j in range(size):
                pc.append(data[0][j*c_size:j*c_size+c_size].view(1,c_size))
                if len(data)>3:
                    idx = np.arange(np.floor(data[3][np.int(j*c_size/5)]), np.floor(data[3][np.int((j*c_size+c_size)/5)]+1))
                    idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                    tmpsc = data[1][idx]
                    xval = np.linspace(0, idx.shape[0]-1, num=c_size)
                    x = np.arange(idx.shape[0])
                    sc_interp = np.interp(xval, x, tmpsc)
                    sc.append(torch.Tensor(sc_interp).view(1,-1))
                else:
                    sc.append(data[1][j*c_size:j*c_size+c_size].view(1,c_size))
                y.append(data[2].view(1,1))

            pc.append(data[0][-c_size:].view(1,c_size))
            if len(data)>3:
                idx = np.arange(np.floor(data[3][np.int((len(data[0])-c_size-2) / 5)]),
                                np.floor(data[3][np.int((len(data[0])-2) / 5)] + 1))
                idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                tmpsc = data[1][idx]
                xval = np.linspace(0, idx.shape[0]-1, num=c_size)
                x = np.arange(idx.shape[0])
                sc_interp = np.interp(xval, x, tmpsc)
                sc.append(torch.Tensor(sc_interp).view(1,-1))
            else:
                sc.append(data[1][-c_size:].view(1,c_size))
            y.append(data[2].view(1,1))

            pc = torch.cat(pc,0)
            sc = torch.cat(sc,0)
            y = torch.cat(y,0).squeeze()
      
            batch[i] = (pc, sc, y)

        return batch

    def overlap(batch, hopSize):
        pc, sc, y = [], [], []
        print(batch)
        for i, data in enumerate(batch):
            j = 0
            while (j+c_size) < len(data[0]-10): # -10: possible dismatch in size between pc & alignment
                pc.append(data[0][j:j+c_size].view(1,c_size))
                if len(data)>3:
                    idx = np.arange(np.floor(data[3][j]), np.floor(data[3][j+c_size]))
                    idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                    tmpsc = data[1][idx]
                    xval = np.linspace(0, idx.shape[0]-1, num=c_size)
                    x = np.arange(idx.shape[0])
                    sc_interp = np.interp(xval, x, tmpsc)
                    sc.append(torch.Tensor(sc_interp).view(1,-1))
                else:
                    sc.append(data[1][j:j+c_size].view(1,c_size))

                y.append(data[2].view(1,1))
                j+=hopSize

            pc.append(data[0][-c_size:].view(1,c_size))
            if len(data) > 3:
                idx = np.arange(np.floor(data[3][len(data[0]) - c_size - 2]), np.floor(data[3][len(data[0]) - 2]))
                idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1

                tmpsc = data[1][idx]
                xval = np.linspace(0, idx.shape[0] - 1, num=c_size)
                x = np.arange(idx.shape[0])
                sc_interp = np.interp(xval, x, tmpsc)
                sc.append(torch.Tensor(sc_interp).view(1, -1))
            else:
                sc.append(data[1][-c_size:].view(1, c_size))
            y.append(data[2].view(1,1))

            pc = torch.cat(pc,0)
            sc = torch.cat(sc,0)
            y = torch.cat(y,0).squeeze()
            batch[i] = (pc, sc, y)

        return batch

    if overlap_flag:
        batch=overlap(batch, 500)
    else:
        batch = non_overlap(batch)

    return torch.utils.data.dataloader.default_collate(batch)

# loss function, calculate the distane between two latent as the rating
def distance_loss(pitch_v, score_v, target):

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred = 1 - cos(pitch_v, score_v)

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

# check_missing_alignedmidi("symphonic")