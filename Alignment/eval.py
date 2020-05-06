from sklearn import metrics
from model import Net
from lib import load_data, load_test_data, Data2Torch, distance_loss, test_collate
import os, torch, json
from torch.autograd import Variable
from functools import partial
import numpy as np
from scipy.stats import pearsonr

from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate_classification(targets, predictions):
    #print(targets.max(),targets.min(),predictions.max(),predictions.min())
    #predictions[predictions>1]=1
    #predictions[predictions<0]=0
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)

    return np.round(r2, decimals=3)#, np.round(accuracy, decimals=3), np.round(corrcoef, decimals=3), np.round(p, decimals=3)

def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (_input) in enumerate(dataloader):
        pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda())
        target = target.view(-1,1)
        pitch_v, score_v = model(pitch.reshape(-1,pitch.shape[-1]), score.reshape(-1,pitch.shape[-1]))
        out = distance_loss(pitch_v, score_v, target.squeeze(1)) [1]
        # print(out, out.shape, torch.mean(out, 0, keepdim=True).data.cpu().numpy().shape)
        all_predictions.extend(torch.mean(out, 0, keepdim=True).data.cpu().numpy())
        all_targets.extend(torch.mean(target.squeeze(1), 0, keepdim=True).data.cpu().numpy())
        #print(out.detach().data.cpu().numpy(),target.detach().data.cpu().numpy())
    #print(len(all_predictions))
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

# DO NOT change the default values if possible
# except during DEBUGGING

def main():
    train_metrics, val_metrics, test_metrics = [], [], []
    test_metrics_AltoSax, test_metrics_BbClarinet, test_metrics_Flute = [], [], []

    trPC, vaPC, SC = load_data(band, feat, midi_op)
    tePC = load_test_data(band, feat)
    teAltoSaxPC = load_test_data(band, feat, 'Alto Saxophone')
    teBbClarinetPC = load_test_data(band, feat, 'Bb Clarinet')
    teFlutePC = load_test_data(band, feat, 'Flute')

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, SC], midi_op), \
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, SC], midi_op), \
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    te_loader = torch.utils.data.DataLoader(Data2Torch([tePC, SC], midi_op),
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)

    te_AltoSax_loader = torch.utils.data.DataLoader(Data2Torch([teAltoSaxPC, SC], midi_op),
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    te_BbClarinet_loader = torch.utils.data.DataLoader(Data2Torch([teBbClarinetPC, SC], midi_op),
                                                    collate_fn=partial(test_collate, [overlap_flag, chunk_size]),
                                                    **kwargs)
    te_Flute_loader = torch.utils.data.DataLoader(Data2Torch([teFlutePC, SC], midi_op),
                                                    collate_fn=partial(test_collate, [overlap_flag, chunk_size]),
                                                    **kwargs)

    eval_metrics = dict()
    for i in range(0,10):
        model_name = '202055/Similarity_batch32_lr0.05_midialigned_s_{}_sample2_chunksize1000_{}_{}{}_score{}_NORM_' \
                     .format(process_collate, model_choose, band, split, score_choose) + str(i)

        #model_name = model_choose + '_' + str(i)
        # if resize the midi to fit the length of audio
        resample = False
        if midi_op == 'resize':
            resample = True

        model_path = './model/'+model_name+'/model'
        model = Net(model_choose)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(model_path)['state_dict'])
        tr = evaluate_model(model, tr_loader)
        va = evaluate_model(model, va_loader)
        te = evaluate_model(model, te_loader)

        te_AltoSax = evaluate_model(model, te_AltoSax_loader)
        te_BbClarinet = evaluate_model(model, te_BbClarinet_loader)
        te_Flute = evaluate_model(model, te_Flute_loader)

        train_metrics.append(tr)
        val_metrics.append(va)
        test_metrics.append(te)

        test_metrics_AltoSax.append(te_AltoSax)
        test_metrics_BbClarinet.append(te_BbClarinet)
        test_metrics_Flute.append(te_Flute)

        print(tr, va, te, te_AltoSax, te_BbClarinet, te_Flute)
        eval_metrics[i] = (tr, va, te, te_AltoSax, te_BbClarinet, te_Flute)

    eval_metrics['avg'] = (
        sum(train_metrics) / len(train_metrics),
        sum(val_metrics) / len(val_metrics),
        sum(test_metrics) / len(test_metrics),
        sum(test_metrics_AltoSax) / len(test_metrics_AltoSax),
        sum(test_metrics_BbClarinet) / len(test_metrics_BbClarinet),
        sum(test_metrics_Flute) / len(test_metrics_Flute)
    )

    model_n = "Similarity_batch32_lr0.05_midialigned_s_{}_sample2_chunksize1000_{}".format(process_collate, model_choose)
    results_dir = './results'
    results_fp = os.path.join(
        results_dir,
        model_n + f'{band}{split}{score_choose}_results_dict.json'
    )
    if not os.path.exists(os.path.dirname(results_fp)):
        os.makedirs(os.path.dirname(results_fp))
    with open(results_fp, 'w') as outfile:
        json.dump(eval_metrics, outfile, indent=2)
    print(len(train_metrics), len(val_metrics), len(test_metrics))
    print('model :', model_name)
    print('train metrics', sum(train_metrics)/len(train_metrics))
    print('valid metrics', sum(val_metrics)/len(val_metrics))
    print('test metrics', sum(test_metrics)/len(test_metrics))

    print('test metrics AltoSax', sum(test_metrics_AltoSax) / len(test_metrics_AltoSax))
    print('test metrics BbClarinet', sum(test_metrics_BbClarinet) / len(test_metrics_BbClarinet))
    print('test metrics Flute', sum(test_metrics_Flute) / len(test_metrics_Flute))

    print('--------------------------------------------------')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # string
    parser.add_argument("--midi_op", type=str, default=midi_op)
    #parser.add_argument("--model_name", type=str, default=model_name, help="model name e.g. 20191028/testmodel")
    parser.add_argument("--model_choose", type=str, default=model_choose)

    # int
    parser.add_argument("--chunk_size", type=int, default=chunk_size)

    # bool
    parser.add_argument("--overlap", type=bool, default=overlap_flag)

    args = parser.parse_args()

    # overwrite params
    #model_name = args.model_name
    midi_op = args.midi_op
    overlap_flag = args.overlap
    model_choose = args.model_choose
    chunk_size = args.chunk_size

    main()