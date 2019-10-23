import os, torch
from datetime import date
from model import Net
from train_utils import Trainer
from lib import load_data, Data2Torch, my_collate, check_missing_alignedmidi
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

band = 'middle'
feat = 'pitch contour'
midi_op = 'resize' # 'sec', 'beat', 'resize', 'aligned', 'aligned_s'

# training parameters
batch_size = 16
num_workers = 2
shuffle = True
epoch = 1000
lr = 0.001
model_name = 'Similarity_batch16_lr0.001_midiResize_windowChunk3sample10sec_CRNN'

print('batch_size: {}, num_workers: {}, epoch: {}, lr: {}, model_name: {}'.format(batch_size, num_workers, epoch, lr, model_name))
print('band: {}, feat: {}, midi_op: {}'.format(band, feat, midi_op))

#check_missing_alignedmidi(band, feat, midi_op)

# model saving path
date = date.today()
out_model_fn = './model/%d%d%d/%s/'%(date.year,date.month,date.day,model_name)
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
tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, SC], midi_op), collate_fn=my_collate, **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, SC], midi_op), collate_fn=my_collate, **t_kwargs)

# build model (function inside model.py)
model = Net()
if torch.cuda.is_available():
    model.cuda()

# start training (function inside train_utils.py)
Trer = Trainer(model, lr, epoch, out_model_fn)
Trer.fit(tr_loader, va_loader)

print(model_name)