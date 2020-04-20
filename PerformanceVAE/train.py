import os
import torch
from functools import partial
import numpy as np
import random
from model import *
from trainer import *
from lib import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # change

from config import *

# DO NOT change the default values if possible
# except during DEBUGGING

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():

    for i in range(0,12):

        manualSeed = i
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        # if you are suing GPU
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        model_name = f'{model_choose}_' \
                     f'batch{batch_size}_' \
                     f'lr{lr}_midi{midi_op}_' \
                     f'{process_collate}_' \
                     f'sample{sample_num}_' \
                     f'chunksize{chunk_size}_' \
                     f'input_type{input_type}' \
                     f'{band}{split}_{manualSeed}'

        print(
            f'batch_size: {batch_size}, num_workers: {num_workers}, epoch: {epoch}, lr: {lr}, model_name: {model_name}'
        )
        print(f'band: {band}, feat: {feat}, midi_op: {midi_op}')

        # model saving path
        from datetime import date
        date = date.today()
        out_model_fn = f'./model/{date.year}{date.month}{date.day}/{model_name}/'
        if not os.path.exists(out_model_fn):
            os.makedirs(out_model_fn)

        # load training and validation data (function inside lib.py)
        trPC, vaPC, SC = load_data(band, feat, midi_op)

        # if resize the midi to fit the length of audio
        resample = False
        if midi_op == 'resize':
            resample = True

        # prepare dataloader (function inside lib.py)
        t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
        v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
        tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, SC], midi_op), worker_init_fn=np.random.seed(manualSeed), \
                                                collate_fn=partial(my_collate, [process_collate, sample_num, chunk_size]), \
                                                **t_kwargs)
        va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, SC], midi_op), worker_init_fn=np.random.seed(manualSeed), \
                                                collate_fn=partial(my_collate, [process_collate, sample_num, chunk_size]), \
                                                **v_kwargs)
  
        # build model (function inside model.py)
        num_in_channels = 1
        if input_type == 'w_score':
            num_in_channels = 2
        model = PCPerformanceEncoder(
            input_size=chunk_size,
            num_in_channels=num_in_channels,
            dropout_prob=dropout_prob,
            z_dim=z_dim,
            kernel_size=kernel_size,
            stride=stride,
            num_rec_layers=num_rec_layers,
            num_conv_features=num_conv_features
        )
        if torch.cuda.is_available():
            model.cuda()    

        # start training (function inside train_utils.py)
        trainer = Trainer(model, lr, epoch, out_model_fn, input_type=input_type, log=log)
        trainer.fit(tr_loader, va_loader)

        print(model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # string
    parser.add_argument("--midi_op", type=str, default=midi_op)
    parser.add_argument("--process_collate", type=str, default=process_collate)

    parser.add_argument("--band", type=str, default=band)
    parser.add_argument("--split", type=str, default=split)
    parser.add_argument("--score_choose", type=str, default=score_choose)

    # int
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--sample_num", type=int, default=sample_num)
    parser.add_argument("--chunk_size", type=int, default=chunk_size)
    parser.add_argument("--num_rec_layers", type=int, default=num_rec_layers)
    parser.add_argument("--z_dim", type=int, default=z_dim)
    parser.add_argument("--kernel_size", type=int, default=kernel_size)
    parser.add_argument("--stride", type=int, default=stride)
    parser.add_argument("--num_conv_features", type=int, default=num_conv_features)
    parser.add_argument("--log", type=int, default=1)

    # float
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--dropout_prob", type=float, default=dropout_prob)

    args = parser.parse_args()

    # overwrite params
    midi_op = args.midi_op
    process_collate = args.process_collate
    band = args.band
    split = args.split
    score_choose = args.score_choose
    batch_size = args.batch_size
    sample_num = args.sample_num
    chunk_size = args.chunk_size
    lr = args.lr
    log = bool(args.log)
    num_rec_layers = args.num_rec_layers
    z_dim = args.z_dim
    kernel_size = args.kernel_size
    stride = args.stride
    num_conv_features = args.num_conv_features

    main()
