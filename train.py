import os, torch
from model import PCConvLstmNet
from train_utils import Trainer
from lib import load_data, Data2Torch

# training parameters
batch_size = 16
num_workers = 2
shuffle = True
epoch = 100
lr = 0.01
out_model_fn = './model/%d%d%d/%s/'%(date.year,date.month,date.day,'RNN_similarity')
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load training and validation data
XtrPC, XtrSC, Ytr, XvaPC, XvaSC, Yva = load_data()

# prepare dataloader
t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch([XtrPC, XtrSC, Ytr]), **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch([XvaPC, XvaSC, Yva]), **t_kwargs)

# build model
PCmodel = PCConvLstmNet().cuda()
SCmodel = PCConvLstmNet().cuda() 

# start training
Trer = Trainer(PCmodel, SCmodel, lr, epoch, out_model_fn)
Trer.fit(tr_loader, va_loader)