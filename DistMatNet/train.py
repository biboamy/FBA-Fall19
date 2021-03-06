import os, torch, random
from model import DistMatNet
from train_utils import Trainer
from lib import load_data, Data2Torch
import numpy as np

from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(4,10):
        manualSeed = i
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        # if you are suing GPU
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        model_name = '{}_{}_batch{}_lr{}_{}_{}_isNorm{}_{}'.format(model_choose, matrix_dim, batch_size, lr, band, split, isNorm, manualSeed)

        print('range: {}, batch_size: {}, num_workers: {}, epoch: {}, lr: {}, model_name: {}'.format(i,batch_size, num_workers, epoch,
                                                                                        lr, model_name))
        print('band: {}, split: {}, matrix_dim: {}'.format(band, split, matrix_dim))

        # model saving path
        from datetime import date
        date = date.today()
        out_model_fn = './model/%d%d%d/%s/' % (date.year, date.month, date.day, model_name)
        if not os.path.exists(out_model_fn):
            os.makedirs(out_model_fn)

        # load training and validation data (function inside lib.py)
        trPC, vaPC = load_data(band)
       
        # prepare dataloader (function inside lib.py)
        t_kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
        v_kwargs = {'batch_size': batch_size, 'pin_memory': True}
        tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), worker_init_fn=np.random.seed(manualSeed), **t_kwargs)
        va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), worker_init_fn=np.random.seed(manualSeed), **v_kwargs)
        
        # build model (function inside model.py)
        model = DistMatNet(model_name)
        model.to(device)
        
        # start training (function inside train_utils.py)
        Trer = Trainer(model, lr, epoch, out_model_fn)
        Trer.fit(tr_loader, va_loader, device)

        print(model_name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--band", type=str, default=band)
    parser.add_argument("--split", type=str, default=split)
    parser.add_argument("--score_choose", type=str, default=score_choose)

    args = parser.parse_args()

    band = args.band
    split = args.split
    score_choose = args.score_choose

    main()