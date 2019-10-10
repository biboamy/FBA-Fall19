from sklearn import metrics
from model import Net
from lib import load_data, load_test_data, Data2Torch, distance_loss
import os, torch
from torch.autograd import Variable
import numpy as np
from scipy.stats import pearsonr

def evaluate_classification(targets, predictions):
    print(targets.max(),targets.min(),predictions.max(),predictions.min())
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)

    return r2, accuracy, corrcoef, p

def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (_input) in enumerate(dataloader):
        pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda())
        target = target.view(-1,1)
        pitch_v, score_v = model(pitch, score)
        out = distance_loss(pitch_v, score_v, target.squeeze(1)) [1]
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(target.data.cpu().numpy())
        #print(out.detach().data.cpu().numpy(),target.detach().data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

band = 'middle'
feat = 'pitch contour'
midi_op = 'resize'
num_workers = 4

# if resize the midi to fit the length of audio
resample = False
if midi_op == 'resize':
    resample = True

trPC, vaPC, SC = load_data(band, feat, 'resize')
tePC = load_test_data(band, feat)

kwargs = {'num_workers': num_workers, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, SC], resample), **kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, SC], resample), **kwargs)
te_loader = torch.utils.data.DataLoader(Data2Torch([tePC, SC], resample), **kwargs)

model_path = './model/2019109/RNN_similarity/model'
model = Net().cuda()
model.load_state_dict(torch.load(model_path)['state_dict'])

train_metrics = evaluate_model(model, tr_loader)
print('train metrics', train_metrics)
val_metrics = evaluate_model(model, va_loader)
print('valid metrics', val_metrics)
test_metrics = evaluate_model(model, te_loader)
print('test metrics', test_metrics)
