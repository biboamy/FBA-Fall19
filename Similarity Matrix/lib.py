import numpy as np
import torch, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_data(path):
    # Load pitch contours
    # Currently only allow pitch contour as feature
    import dill

    # train
    trPC = np.array(dill.load(open(path + 'test.dill', 'rb')))
    # valid
    vaPC = np.array(dill.load(open(path + 'test.dill', 'rb')))

    return trPC, vaPC

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
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        oup = self.data[0][index]['matrix']
        score = self.data[0][index]['ratings'][0] #choose which score

        return torch.from_numpy(oup).float(), torch.from_numpy(np.array([score])).float()
    
    def __len__(self):
        return len(self.data)

# loss function, calculate the distane between two latent as the rating
def loss_func(pred, target):

    mse = nn.MSELoss()
    loss = mse(pred, target)

    return loss
