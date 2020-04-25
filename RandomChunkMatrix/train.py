import os, torch, random, h5py, dill
from model import Net_Fixed
from train_utils import Trainer
from functools import partial
from lib import load_data, Data2Torch, my_collate
import numpy as np

from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # change

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():

    # load training and validation data (function inside lib.py)
    trPC, vaPC, SC = load_data(band)

    # load the original matrices .h5 file and id2idx file
    orig_mtx_h5 = h5py.File(PATH_FBA_MTX + mtx_orig_h5.format(band), 'r')
    id2idx = dill.load(open(PATH_FBA_MTX + id2idx_file.format(band), 'rb'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(0,1): # 10
        manualSeed = i
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        # if you are suing GPU
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        model_name = 'ChunkedInput_{}_{}_batch{}_lr{}_{}_{}_isNorm{}_{}'.format(model_choose, chunk_matrix_dim, batch_size, lr, band, split, isNorm, manualSeed)

        print('range: {}, batch_size: {}, num_workers: {}, epoch: {}, lr: {}, model_name: {}'.format(i, batch_size, num_workers, epoch,
                                                                                        lr, model_name))
        print('band: {}, split: {}, chunk_matrix_dim: {}'.format(band, split, chunk_matrix_dim))

        # model saving path
        from datetime import date
        date = date.today()
        out_model_fn = './model/%d%d%d/%s/' % (date.year, date.month, date.day, model_name)
        if not os.path.exists(out_model_fn):
            os.makedirs(out_model_fn)

        # prepare dataloader (function inside lib.py)
        t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,
                    'drop_last': True}
        v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
        tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, id2idx, orig_mtx_h5, SC]),worker_init_fn=np.random.seed(manualSeed), \
                                                collate_fn=partial(my_collate,[process_collate, sample_num, chunk_size]), \
                                                **t_kwargs)
        va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, id2idx, orig_mtx_h5, SC]),worker_init_fn=np.random.seed(manualSeed), \
                                                collate_fn=partial(my_collate, [process_collate, sample_num, chunk_size]), \
                                                **v_kwargs)


        # build model (function inside model.py)
        model = Net_Fixed(model_name)
        model.to(device)
        
        # start training (function inside train_utils.py)
        Trer = Trainer(model, lr, epoch, out_model_fn)
        Trer.fit(tr_loader, va_loader, device)

        print(model_name)

    orig_mtx_h5.close()

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