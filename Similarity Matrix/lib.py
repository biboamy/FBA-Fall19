import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import h5py
import dill

def load_data(matrix_path, band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature

    # train
    trPC = dill.load(open(matrix_path + 'matrix_fixed_train.dill', 'rb'))
    # valid
    vaPC = dill.load(open(matrix_path + 'matrix_fixed_valid.dill', 'rb'))

    return trPC, vaPC

def load_test_data(matrix_path, band='middle'):
    # Load pitch contours
    # Currently only allow pitch contour as feature

    # Read features from .dill files
    tePC = dill.load(open(matrix_path + 'matrix_fixed_test.dill', 'rb'))

    return tePC

class Data2Torch(Dataset):
    def __init__(self, data):
        self.data = data[0]

    def __getitem__(self, index):
        oup = self.data[index]['matrix']
        score = self.data[index]['ratings'][0]

        return torch.from_numpy(oup).float(), torch.from_numpy(np.array([score])).float()
    
    def __len__(self):
        return len(self.data)

# loss function, calculate the distane between two latent as the rating
def loss_func(pred, target):

    mse = nn.MSELoss()
    loss = mse(pred, target)

    return loss
