from sklearn import metrics
from model import Net, Net_Fixed
from lib import load_data, load_test_data, Data2Torch
import os, torch
from torch.autograd import Variable
from functools import partial
import numpy as np
from scipy.stats import pearsonr

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

band = 'middle'
num_workers = 4
model_choose = 'ConvNet_Fixed'

model_name = '20191120/ConvNet_Fixed_batch16_lr1e-05'

def main():

    matrix_path = '../../../data_share/FBA/fall19/data/matrix/'
    trPC, vaPC = load_data(matrix_path)
    tePC = load_test_data(matrix_path)

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), **kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), **kwargs)
    te_loader = torch.utils.data.DataLoader(Data2Torch([tePC]), **kwargs)

    model_path = './model/'+model_name+'/model'
    # build model (function inside model.py)
    if model_choose == 'ConvNet':
        model = Net()
    elif model_choose == 'ConvNet_Fixed':
        model = Net_Fixed()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'])

    print('model :', model_name)
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
    parser.add_argument("--model_name", type=str, default=model_name, help="model name e.g. 20191028/testmodel")
    parser.add_argument("--model_choose", type=str, default=model_choose)

    args = parser.parse_args()

    # overwrite params
    model_name = args.model_name
    model_choose = args.model_choose

    print(model_name, model_choose)

    main()