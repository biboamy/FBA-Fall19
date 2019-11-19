import os, torch
from model import Net
from train_utils import Trainer
from functools import partial
from lib import load_data, Data2Torch, my_collate, check_missing_alignedmidi
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

# DO NOT change the default values if possible
# except during DEBUGGING

band = 'middle' # fixed
feat = 'pitch contour' # fixed
midi_op = 'aligned_s' # 'sec', 'beat', 'resize', 'aligned', 'aligned_s'
model_choose = 'CNN'

# training parameters
batch_size = 16
num_workers = 2 # fixed
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.001

loss_func = 'Similarity'
process_collate = 'randomChunk' # 'randomChunk', 'windowChunk', 'padding'
sample_num = 3 # numbers of chunks # if choosing windowChunk, sample_num has to be 1
chunk_size = 1000 # 1000 ~ 5 sec / 2000 ~ 10 sec

def main():

    model_name = '{}_batch{}_lr{}_midi{}_{}_sample{}_chunksize{}_{}'.format(loss_func, batch_size, lr, midi_op, \
                                                                            process_collate, sample_num, chunk_size, \
                                                                            model_choose)
    #'Similarity_batch16_lr0.001_midialigneds_windowChunk1sample10sec_CNN'

    print('batch_size: {}, num_workers: {}, epoch: {}, lr: {}, model_name: {}'.format(batch_size, num_workers, epoch, lr, model_name))
    print('band: {}, feat: {}, midi_op: {}'.format(band, feat, midi_op))

    #check_missing_alignedmidi(band, feat, midi_op)

    # model saving path
    from datetime import date
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
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC, SC], midi_op), \
                                            collate_fn=partial(my_collate, [process_collate, sample_num, chunk_size]), \
                                            **t_kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC, SC], midi_op), \
                                            collate_fn=partial(my_collate, [process_collate, sample_num, chunk_size]), \
                                            **t_kwargs)

    # build model (function inside model.py)
    model = Net(model_choose)
    if torch.cuda.is_available():
        model.cuda()

    # start training (function inside train_utils.py)
    Trer = Trainer(model, lr, epoch, out_model_fn)
    Trer.fit(tr_loader, va_loader)

    print(model_name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # string
    parser.add_argument("--loss_func", type=str, default=loss_func)
    parser.add_argument("--midi_op", type=str, default=midi_op)
    parser.add_argument("--process_collate", type=str, default=process_collate)
    parser.add_argument("--model_choose", type=str, default=model_choose)

    # int
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--sample_num", type=int, default=sample_num)
    parser.add_argument("--chunk_size", type=int, default=chunk_size)

    # float
    parser.add_argument("--lr", type=float, default=lr)

    args = parser.parse_args()

    # overwrite params
    loss_func = args.loss_func
    midi_op = args.midi_op
    process_collate = args.process_collate
    model_choose = args.model_choose
    batch_size = args.batch_size
    sample_num = args.sample_num
    chunk_size = args.chunk_size
    lr = args.lr

    main()