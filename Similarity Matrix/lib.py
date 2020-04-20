import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dill

from config import *

def load_data(band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature

    # train
    trPC = dill.load(open(PATH_FBA_MTX + data_train_mtx[split].format(band, matrix_dim), 'rb'))
    # valid
    vaPC = dill.load(open(PATH_FBA_MTX + data_valid_mtx[split].format(band, matrix_dim), 'rb'))

    return trPC, vaPC

def load_test_data(band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature

    # Read features from .dill files
    tePC = dill.load(open(PATH_FBA_MTX + data_test_mtx[split].format(band, matrix_dim), 'rb'))

    return tePC

class Data2Torch(Dataset):
    def __init__(self, data):
        self.data = data[0]

    def __getitem__(self, index):
        oup = self.data[index]['matrix']
        score = self.data[index]['ratings'][score_choose]

        if isNorm:
            oup = oup/7

        return torch.from_numpy(oup).float(), torch.from_numpy(np.array([score])).float()
    
    def __len__(self):
        return len(self.data)

# loss function, calculate the distane between two latent as the rating
def loss_func(pred, target):

    mse = nn.MSELoss()
    loss = mse(pred, target)

    return loss