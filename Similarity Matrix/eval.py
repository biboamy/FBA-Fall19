from sklearn import metrics
from model import Net, Net_Fixed
from lib import load_data, load_test_data, Data2Torch
import os, torch, random
from torch.autograd import Variable
from functools import partial
import numpy as np
from scipy.stats import pearsonr
from config import *

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def evaluate_classification(targets, predictions):
    print(targets.max(),targets.min(),predictions.max(),predictions.min())
    predictions[predictions>1]=1
    predictions[predictions<0]=0
    #print(np.squeeze(targets), predictions)
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), np.squeeze(predictions))
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)

    return np.round(r2, decimals=3), np.round(accuracy, decimals=3), np.round(corrcoef, decimals=3), np.round(p, decimals=3)

def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (_input) in enumerate(dataloader):
        matrix, target = Variable(_input[0].cuda()), Variable(_input[1].cuda())
        target = target.view(-1,1)
        pred = model(matrix.unsqueeze(1))
        #print(target, pred)
        all_predictions.extend(torch.mean(pred, 0, keepdim=True).data.cpu().numpy())
        all_targets.extend(torch.mean(target, 0, keepdim=True).data.cpu().numpy())
        #print(out.detach().data.cpu().numpy(),target.detach().data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

# DO NOT change the default values if possible
# except during DEBUGGING

def main():

    matrix_path = '../../../data_share/FBA/fall19/data/matrix/'
    trPC, vaPC = load_data(matrix_path)
    tePC = load_test_data(matrix_path)

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), worker_init_fn=np.random.seed(manualSeed), **kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), worker_init_fn=np.random.seed(manualSeed), **kwargs)
    te_loader = torch.utils.data.DataLoader(Data2Torch([tePC]), worker_init_fn=np.random.seed(manualSeed), **kwargs)

    model_path = './model/'+model_name_e+'/model'
    # build model (function inside model.py)
    if model_choose == 'ConvNet':
        model = Net()
    elif model_choose == 'ConvNet_Fixed':
        model = Net_Fixed()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'])

    print('model :', model_name_e)
    train_metrics = evaluate_model(model, tr_loader)
    print('train metrics', train_metrics)
    val_metrics = evaluate_model(model, va_loader)
    print('valid metrics', val_metrics)
    test_metrics = evaluate_model(model, te_loader)
    print('test metrics', test_metrics)
    print('--------------------------------------------------')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # string
    parser.add_argument("--model_name_e", type=str, default=model_name_e, help="model name e.g. 20191028/testmodel")
    parser.add_argument("--model_choose", type=str, default=model_choose)

    args = parser.parse_args()

    # overwrite params
    model_name_e = args.model_name_e
    model_choose = args.model_choose

    print(model_name_e, model_choose)

    main()