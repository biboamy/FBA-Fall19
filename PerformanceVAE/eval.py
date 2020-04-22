from sklearn import metrics
import os
import torch
import random
from torch.autograd import Variable
from functools import partial
import numpy as np
from scipy.stats import pearsonr
from model import *
from lib import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


def evaluate_classification(targets, predictions):
    print(targets.max(), targets.min(), predictions.max(), predictions.min(), len(predictions))
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0
    # print(np.squeeze(targets), predictions)
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), np.squeeze(predictions))
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)

    return np.round(r2, decimals=3), np.round(accuracy, decimals=3), np.round(corrcoef, decimals=3), np.round(p, decimals=3)


def evaluate_model(model, dataloader, input_type='w_score'):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (_input) in enumerate(dataloader):
        pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda())
        pitch = normalize_pitch(
            pitch.view(pitch.shape[0] * pitch.shape[1], -1).unsqueeze(1)
        )
        score = normalize_midi(
            score.view(score.shape[0] * score.shape[1], -1).unsqueeze(1)
        )
        if type(model) == PCPerformanceVAE:
            pred = model(pitch)[1]
            all_predictions.extend(pred.squeeze(1).data.cpu().numpy())
        elif type(model) == PCPerformanceEncoder:
            input_tensor = pitch
            if input_type == 'w_score':
                input_tensor = torch.cat((input_tensor, score), 1)
            pred = model(input_tensor).reshape(-1)
            all_predictions.extend(pred.data.cpu().numpy())
        all_targets.extend(target.reshape(-1).data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))


def main():
    train_metrics, val_metrics, test_metrics = [], [], []
    model_n = '2020421/'
    model_n += f'{model_choose}_' \
                  f'batch{batch_size}_' \
                  f'lr{lr}_midi{midi_op}_' \
                  f'{process_collate}_' \
                  f'sample{sample_num}_' \
                  f'chunksize{chunk_size}_' \
                  f'input_type{input_type}'
    trPC, vaPC, SC = load_data(band, feat, midi_op)
    tePC = load_test_data(band, feat)

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(
        Data2Torch([trPC, SC], midi_op), collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs
    )
    va_loader = torch.utils.data.DataLoader(
        Data2Torch([vaPC, SC], midi_op), collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs
    )
    te_loader = torch.utils.data.DataLoader(
        Data2Torch([tePC, SC], midi_op), collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs
    )
    for i in range(0, 12):
        model_name = model_n + f'{band}{split}_{i}'

        # if resize the midi to fit the length of audio
        resample = False
        if midi_op == 'resize':
            resample = True

        model_path = './model/'+model_name+'/model'
        num_in_channels = 1
        if input_type == 'w_score':
            num_in_channels = 2
        if model_choose == 'PerformanceVAE':
            model = PCPerformanceVAE(
                input_size=chunk_size,
                num_in_channels=num_in_channels,
                dropout_prob=dropout_prob,
                z_dim=z_dim,
                kernel_size=kernel_size,
                stride=stride,
                num_conv_features=num_conv_features
            )
        elif model_choose == 'PerformanceEncoder':
            model = PCPerformanceEncoder(
                input_size=chunk_size,
                num_in_channels=num_in_channels,
                dropout_prob=dropout_prob,
                z_dim=z_dim,
                kernel_size=kernel_size,
                stride=stride,
                num_conv_features=num_conv_features
            )
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(model_path)['state_dict'])
        tr = evaluate_model(model, tr_loader)[0]
        va = evaluate_model(model, va_loader)[0]
        te = evaluate_model(model, te_loader)[0]
        train_metrics.append(tr)
        val_metrics.append(va)
        test_metrics.append(te)
        print(tr, va, te)
        del model

    print('model :', model_n)
    print('train metrics', sum(train_metrics)/len(train_metrics))
    print('valid metrics', sum(val_metrics)/len(val_metrics))
    print('test metrics', sum(test_metrics)/len(test_metrics))
    print('--------------------------------------------------')


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
    parser.add_argument("--log", type=int, default=log)

    # float
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--beta", type=float, default=beta)
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
    beta = args.beta
    num_rec_layers = args.num_rec_layers
    z_dim = args.z_dim
    kernel_size = args.kernel_size
    stride = args.stride
    num_conv_features = args.num_conv_features

    main()