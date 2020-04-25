from sklearn import metrics
from model import Net_Fixed
from lib import load_data, load_test_data, Data2Torch, test_collate
from functools import partial
import os, torch, h5py, dill
from torch.autograd import Variable
import numpy as np
from scipy.stats import pearsonr
import statistics

from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

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

    return np.round(r2, decimals=3), np.round(accuracy, decimals=3), np.round(corrcoef, decimals=3), np.round(p, decimals=3)

def evaluate_model(model, dataloader):
    all_predictions = []
    all_targets = []
    for i, (_input) in enumerate(dataloader):
        matrix, target = Variable(_input[0].cuda()), Variable(_input[1].cuda())
        pred = model(matrix.view(-1,1,matrix.shape[2], matrix.shape[3]))
        all_predictions.extend(pred.squeeze(1).data.cpu().numpy())
        all_targets.extend(target.view(-1,1).data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

# DO NOT change the default values if possible
# except during DEBUGGING

def main(model_name_e):

    trPC, vaPC, SC = load_data(band)
    tePC = load_test_data(band)

    # load the original matrices .h5 file and id2idx file
    orig_mtx_h5 = h5py.File(PATH_FBA_MTX + mtx_orig_h5.format(band), 'r')
    id2idx = dill.load(open(PATH_FBA_MTX + id2idx_file.format(band), 'rb'))

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, id2idx, orig_mtx_h5, SC]), \
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, id2idx, orig_mtx_h5, SC]), \
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    te_loader = torch.utils.data.DataLoader(Data2Torch([tePC, id2idx, orig_mtx_h5, SC]),
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    tr = []
    va = []
    te = []

    print(model_name_e)

    for i in range(0, 1):
        if True:
            model_name = model_name_e+'_'+str(i)

            model_path = './model/'+model_name+'/model'
            # build model (function inside model.py)
            model = Net_Fixed(model_name)
            if torch.cuda.is_available():
                model.cuda()
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.eval()

            for i in [1, 4, 7]:
                model.model.conv[i].bn1.momentum = 0
                model.model.conv[i].bn2.momentum = 0
                model.model.conv[i].bn3.momentum = 0
                model.model.conv[i].bn1.track_running_stats = False
                model.model.conv[i].bn2.track_running_stats = False
                model.model.conv[i].bn3.track_running_stats = False

            print('model :', model_name)
            train_metrics = evaluate_model(model, tr_loader)
            print('train metrics', train_metrics)
            val_metrics = evaluate_model(model, va_loader)
            print('valid metrics', val_metrics)
            test_metrics = evaluate_model(model, te_loader)
            print('test metrics', test_metrics)
            print('--------------------------------------------------')

            tr.extend([train_metrics[0]])
            va.extend([val_metrics[0]])
            te.extend([test_metrics[0]])
    print(tr, max(tr), min(tr), statistics.median(tr))
    print(va, max(va), min(va), statistics.median(va))
    print(te, max(te), min(te), statistics.median(te))
    print("{:.3f}, {:.3f}, {:.3f}" .format(sum(tr)/len(tr),sum(va)/len(va),sum(te)/len(te)))

    orig_mtx_h5.close()

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