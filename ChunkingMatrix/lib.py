import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import h5py
import dill
from config import *


def load_data(matrix_path, band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature

    # train
    trPC = dill.load(open(matrix_path + 'matrix_chunk{}_train{}.dill'.format(sub_num, sub_matrix_dim), 'rb'))
    # valid
    vaPC = dill.load(open(matrix_path + 'matrix_chunk{}_valid{}.dill'.format(sub_num, sub_matrix_dim), 'rb'))

    return trPC, vaPC


def load_test_data(matrix_path, band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature

    # Read features from .dill files
    tePC = dill.load(open(matrix_path + 'matrix_chunk{}_test{}.dill'.format(sub_num, sub_matrix_dim), 'rb'))

    return tePC


class Data2Torch(Dataset):
    def __init__(self, data, lstm=False):
        self.data = data[0]
        self.lstm = lstm

    def __getitem__(self, index):
        if self.lstm == False:
            idx = int(index/sub_num)
            rem = index - idx * sub_num
            oup = self.data[idx]['matrix'][rem]
        else:
            oup = self.data[index]['matrix']
        score = self.data[index]['ratings'][score_choose]

        return torch.from_numpy(oup).float(), torch.from_numpy(np.array([score])).float()

    def __len__(self):
        if self.lstm == False:
            return len(self.data) * sub_num
        else:
            return len(self.data)


# loss function, calculate the distane between two latent as the rating
def loss_func(pred, target):
    mse = nn.MSELoss()
    loss = mse(pred, target)

    return loss