from sklearn import metrics
from model import Net
from lib import load_data, load_test_data, Data2Torch, distance_loss, test_collate
import os, torch
from torch.autograd import Variable
from functools import partial
import numpy as np
from scipy.stats import pearsonr

def evaluate_classification(targets, predictions):
    print(targets.max(),targets.min(),predictions.max(),predictions.min())
    predictions[predictions>1]=1
    predictions[predictions<0]=0
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), predictions)
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
        pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda())
        target = target.view(-1,1)
        pitch_v, score_v = model(pitch.reshape(-1,pitch.shape[-1]), score.reshape(-1,pitch.shape[-1]))
        out = distance_loss(pitch_v, score_v, target.squeeze(1)) [1]
        all_predictions.extend(torch.mean(out, 0, keepdim=True).data.cpu().numpy())
        all_targets.extend(torch.mean(target, 0, keepdim=True).data.cpu().numpy())
        #print(out.detach().data.cpu().numpy(),target.detach().data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

# DO NOT change the default values if possible
# except during DEBUGGING

band = 'middle'
feat = 'pitch contour'
midi_op = 'aligned_s' # 'sec', 'beat', 'resize', 'aligned', 'aligned_s'
num_workers = 4
model_choose = 'CNN'

overlap_flag = False
chunk_size = 1000
model_name = '20191028/Similarity_batch16_lr0.001_midialigned_s_randomChunk_sample3_chunksize2000_CNN'

def main():

    # if resize the midi to fit the length of audio
    resample = False
    if midi_op == 'resize':
        resample = True

    trPC, vaPC, SC = load_data(band, feat, midi_op)
    tePC = load_test_data(band, feat)

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, SC], midi_op), \
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, SC], midi_op), \
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)
    te_loader = torch.utils.data.DataLoader(Data2Torch([tePC, SC], midi_op),
                                            collate_fn=partial(test_collate, [overlap_flag, chunk_size]), **kwargs)

    model_path = './model/'+model_name+'/model'
    model = Net(model_choose)
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
    parser.add_argument("--midi_op", type=str, default=midi_op)
    parser.add_argument("--model_name", type=str, default=model_name, help="model name e.g. 20191028/testmodel")
    parser.add_argument("--model_choose", type=str, default=model_choose)

    # int
    parser.add_argument("--chunk_size", type=int, default=chunk_size)

    # bool
    parser.add_argument("--overlap", type=bool, default=overlap_flag)

    args = parser.parse_args()

    # overwrite params
    model_name = args.model_name
    midi_op = args.midi_op
    overlap_flag = args.overlap
    model_choose = args.model_choose
    chunk_size = args.chunk_size

    print(model_name, midi_op, overlap_flag, model_choose, chunk_size)

    main()