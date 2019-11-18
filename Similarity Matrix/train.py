import os, torch
from model import Net
from train_utils import Trainer
from lib import load_data, Data2Torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

# training parameters
batch_size = 1
num_workers = 4 # fixed
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.001

def main():

    model_name = 'test_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model saving path
    from datetime import date
    date = date.today()
    out_model_fn = './model/'+(model_name)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    # load training and validation data (function inside lib.py)
    trPC, vaPC = load_data('./../data_share/FBA/fall19/data/matrix/middle_2_3_matrix.h5')
    
    # prepare dataloader (function inside lib.py)
    t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
    v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), **t_kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), **t_kwargs)
    
    # build model (function inside model.py)
    model = Net()
    model.to(device)
    
    # start training (function inside train_utils.py)
    Trer = Trainer(model, lr, epoch, out_model_fn)
    Trer.fit(tr_loader, va_loader, device)

    print(model_name)

if __name__ == "__main__":

    main()