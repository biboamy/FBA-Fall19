import torch
import torch.nn as nn

def load_data():



    return XtrPC, XtrSC, Ytr, XvaPC, XvaSC, Yva

class Data2Torch(Dataset):
    def __init__(self, data):
        self.xPC = data[0]
        self.xSC = data[1]
        self.Y = data[2]

    def __getitem__(self, index):

        mXPC = torch.from_numpy(self.xPC[index]).float()
        mXSC = torch.from_numpy(self.xSC[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()
        return mXPC, mXSC, mY
    
    def __len__(self):
        return len(self.X)

def distance_loss(pitch_v, score_v, target):

	pdist = nn.PairwiseDistance(p=2)
	pred = pdist(pitch_v, score_v)

	loss_func = nn.MSELoss()
	loss = loss_func(pred, target)

	return loss

def classify_loss(pitch_v, score_v, target):

	return loss