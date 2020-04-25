import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from skimage import io, transform
import dill

from config import *

# remember to normalize!

def load_data(band='middle', feat='pitch contour', midi_op='res12'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill
    assert (feat == 'pitch contour')

    # train
    trPC = np.array(dill.load(open(PATH_FBA_SPLIT + data_train_pc[split].format(band), 'rb')))
    # valid
    vaPC = np.array(dill.load(open(PATH_FBA_SPLIT + data_valid_pc[split].format(band), 'rb')))

    # Read scores from .dill files
    mid_file = PATH_FBA_MIDI + midi_aligned_s.format(band)
    SC = dill.load(open(mid_file, 'rb'))  # all scores / aligned midi

    return trPC, vaPC, SC


def load_test_data(band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill

    # test
    tePC = np.array(dill.load(open(PATH_FBA_SPLIT + data_test_pc[split].format(band), 'rb')))

    return tePC


class Data2Torch(Dataset):
    def __init__(self, data, midi_op='aligned_s'):
        self.xPC = data[0] # perf
        self.id2idx = data[1] # id2idx
        self.orig_mtx = data[2] # orig_mtx_h5
        self.midi_op = midi_op
        self.resample = False

        self.xSC = data[3]['scores']
        self.align = data[3]['alignment']

        assert midi_op == 'aligned_s'

    def __getitem__(self, index):

        # pitch contour
        PC = self.xPC[index]['pitch_contour']
        mXPC = torch.from_numpy(PC).float()
        # ratings
        mY = torch.from_numpy(np.array([i for i in self.xPC[index]['ratings']])).float()
        mY = mY[score_choose]  # ratting order (0: musicality, 1: note accuracy, 2: rhythmetic, 3: tone quality)

        if self.midi_op == 'aligned_s':
            year = self.xPC[index]['year']
            instrument = self.xPC[index]['instrumemt']
            id = self.xPC[index]['student_id']

            #  score (not used here, just to verify)
            SC = self.xSC[instrument][year]
            SC = np.argmax(SC, axis=0)
            mXSC = torch.from_numpy(SC).float()

            # alignment
            align = self.align[year][id]

            # original matrix
            idx = self.id2idx[year][id]
            mtx_st, mtx_ed, mtx_pc = self.orig_mtx['sc_idx'][idx]
            mXMTX = self.orig_mtx['matrix'][mtx_st:mtx_ed, :mtx_pc]

            if isNorm:
                mXMTX = mXMTX / 7

            oup = [mXPC, mXMTX, mY, align]
        else:
            raise ValueError('Please input the correct model')

        return oup

    def __len__(self):
        return len(self.xPC)


# padding each sequence in the batch to the same length
def my_collate(collate_params, batch):
    process_collate, sample_num, chunk_size = collate_params

    c_size = chunk_size

    def random_chunk(batch):
        import random
        num = sample_num

        for i, data in enumerate(batch):
            mtx = []
            for j in range(num):
                start = round(random.uniform(0, len(
                    data[0]) - c_size - 10))  # -10: possible dismatch in size between pc & alignment

                st_pc = np.int(start / 5)
                st_sc = np.int(data[3][st_pc])
                ed_pc = np.int((start + c_size) / 5)
                ed_sc = np.min([np.int(data[3][ed_pc] + 1), data[1].shape[1] - 1])

                sub_mtx = data[1][st_sc:ed_sc, st_pc:ed_pc].astype(np.float32)

                sub_mtx = transform.resize(sub_mtx, (chunk_matrix_dim, chunk_matrix_dim))

                mtx.append(torch.Tensor(sub_mtx).view(1, chunk_matrix_dim, chunk_matrix_dim))


                batch[i] = (torch.cat(mtx, 0), data[2].repeat(num))

        return batch

    assert process_collate == 'randomChunk'
    batch = random_chunk(batch)

    return torch.utils.data.dataloader.default_collate(batch)


def test_collate(collate_params, batch):
    overlap_flag, chunk_size = collate_params

    c_size = chunk_size

    def non_overlap(batch):
        mtx, y = [], []
        for i, data in enumerate(batch):
            size = int((len(data[0]) - 10) / c_size)  # -10: possible dismatch in size between pc & alignment
            for j in range(size):
                start = j*c_size

                st_pc = np.int(start / 5)
                st_sc = np.int(data[3][st_pc])
                ed_pc = np.int((start + c_size) / 5)
                ed_sc = np.min([np.int(data[3][ed_pc] + 1), data[1].shape[1] - 1])

                sub_mtx = data[1][st_sc:ed_sc, st_pc:ed_pc].astype(np.float32)

                sub_mtx = transform.resize(sub_mtx, (chunk_matrix_dim, chunk_matrix_dim))

                mtx.append(torch.Tensor(sub_mtx).view(1, chunk_matrix_dim, chunk_matrix_dim))

                y.append(data[2].view(1,1))
            #     pc.append(data[0][j * c_size:j * c_size + c_size].view(1, c_size))
            #     if len(data) > 3:
            #         idx = np.arange(np.floor(data[3][np.int(j * c_size / 5)]),
            #                         np.floor(data[3][np.int((j * c_size + c_size) / 5)] + 1))
            #         idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1
            #
            #         tmpsc = data[1][idx]
            #         xval = np.linspace(0, idx.shape[0] - 1, num=c_size)
            #         x = np.arange(idx.shape[0])
            #         sc_interp = np.interp(xval, x, tmpsc)
            #         sc.append(torch.Tensor(sc_interp).view(1, -1))
            #     else:
            #         sc.append(data[1][j * c_size:j * c_size + c_size].view(1, c_size))
            #     y.append(data[2].view(1, 1))
            #

            start = len(data[0])-c_size-2
            st_pc = np.int(start / 5)
            st_sc = np.int(data[3][st_pc])
            ed_pc = np.int((start + c_size) / 5)
            ed_sc = np.min([np.int(data[3][ed_pc] + 1), data[1].shape[1] - 1])

            sub_mtx = data[1][st_sc:ed_sc, st_pc:ed_pc].astype(np.float32)

            sub_mtx = transform.resize(sub_mtx, (chunk_matrix_dim, chunk_matrix_dim))

            mtx.append(torch.Tensor(sub_mtx).view(1, chunk_matrix_dim, chunk_matrix_dim))

            y.append(data[2].view(1, 1))

            # pc.append(data[0][-c_size:].view(1, c_size))
            # if len(data) > 3:
            #     idx = np.arange(np.floor(data[3][np.int((len(data[0]) - c_size - 2) / 5)]),
            #                     np.floor(data[3][np.int((len(data[0]) - 2) / 5)] + 1))
            #     idx[idx >= data[1].shape[0]] = data[1].shape[0] - 1
            #
            #     tmpsc = data[1][idx]
            #     xval = np.linspace(0, idx.shape[0] - 1, num=c_size)
            #     x = np.arange(idx.shape[0])
            #     sc_interp = np.interp(xval, x, tmpsc)
            #     sc.append(torch.Tensor(sc_interp).view(1, -1))
            # else:
            #     sc.append(data[1][-c_size:].view(1, c_size))
            # y.append(data[2].view(1, 1))
            #
            # pc = torch.cat(pc, 0)
            # sc = torch.cat(sc, 0)
            # y = torch.cat(y, 0).squeeze()

            mtx = torch.cat(mtx, 0)
            y = torch.cat(y, 0).squeeze()

            batch[i] = (mtx, y)

        return batch

    batch = non_overlap(batch)

    return torch.utils.data.dataloader.default_collate(batch)
    
    def __len__(self):
        return len(self.data)

# loss function, calculate the distane between two latent as the rating
def loss_func(pred, target):

    mse = nn.MSELoss()
    loss = mse(pred, target)

    return loss