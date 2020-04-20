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

