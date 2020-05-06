from sklearn import metrics
from model import Net_Fixed
from lib import load_data, load_test_data, Data2Torch
import os, torch
from torch.autograd import Variable
import numpy as np
from scipy.stats import pearsonr
import statistics
import json
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

def evaluate_classification(targets, predictions):
    print(targets.max(),targets.min(),predictions.max(),predictions.min(), len(predictions))
    #predictions[predictions>1]=1
    #predictions[predictions<0]=0
    #print(np.squeeze(targets), predictions)
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), np.squeeze(predictions))
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)

    return np.round(r2, decimals=3) #, np.round(accuracy, decimals=3), np.round(corrcoef, decimals=3), np.round(p, decimals=3)

def evaluate_model(model, dataloader):
    all_predictions = []
    all_targets = []
    for i, (_input) in enumerate(dataloader):
        matrix, target = Variable(_input[0].cuda()), Variable(_input[1].cuda())
        pred = model(matrix.unsqueeze(1))
        all_predictions.extend(pred.squeeze(1).data.cpu().numpy())
        all_targets.extend(target.data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

# DO NOT change the default values if possible
# except during DEBUGGING

def main(model_name_e):

    train_metrics, val_metrics, test_metrics = [], [], []
    test_metrics_AltoSax, test_metrics_BbClarinet, test_metrics_Flute = [], [], []

    trPC, vaPC = load_data(band)
    tePC = load_test_data(band)
    teAltoSaxPC = load_test_data(band, 'Alto Saxophone')
    teBbClarinetPC = load_test_data(band, 'Bb Clarinet')
    teFlutePC = load_test_data(band, 'Flute')

    kwargs = {'batch_size': batch_size, 'pin_memory': True}
    #kwargs = {'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), **kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), **kwargs)
    te_loader = torch.utils.data.DataLoader(Data2Torch([tePC]), **kwargs)

    te_AltoSax_loader = torch.utils.data.DataLoader(Data2Torch([teAltoSaxPC]), **kwargs)
    te_BbClarinet_loader = torch.utils.data.DataLoader(Data2Torch([teBbClarinetPC]), **kwargs)
    te_Flute_loader = torch.utils.data.DataLoader(Data2Torch([teFlutePC]), **kwargs)

    print(model_name_e)
    result = {}

    eval_metrics = dict()
    for i in range(0, 10):
        if True:
            model_name = model_name_e+'_'+str(i)

            model_path = './model/'+model_name+'/model'
            # build model (function inside model.py)
            model = Net_Fixed(model_name)
            if torch.cuda.is_available():
                model.cuda()
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.eval()

            for j in [1, 4, 7]:
                model.model.conv[j].bn1.momentum = 0
                model.model.conv[j].bn2.momentum = 0
                model.model.conv[j].bn3.momentum = 0
                model.model.conv[j].bn1.track_running_stats = False
                model.model.conv[j].bn2.track_running_stats = False
                model.model.conv[j].bn3.track_running_stats = False

            print('model :', model_name)
            tr = evaluate_model(model, tr_loader)
            print('train metrics', train_metrics)
            va = evaluate_model(model, va_loader)
            print('valid metrics', val_metrics)
            te = evaluate_model(model, te_loader)
            print('test metrics', test_metrics)
            print('--------------------------------------------------')
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

    print("{:.3f}, {:.3f}, {:.3f}" .format(sum(train_metrics) / len(train_metrics), \
                                           sum(val_metrics) / len(val_metrics), sum(test_metrics) / len(test_metrics)))

    with open('result/'+model_name_e.split('/')[1]+'.json', 'w') as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # string
    parser.add_argument("--model_name_e", type=str, help="model name e.g. 20191028/testmodel")

    args = parser.parse_args()

    # overwrite params
    model_name_e = args.model_name_e

    main(model_name_e)

    print(model_name_e)