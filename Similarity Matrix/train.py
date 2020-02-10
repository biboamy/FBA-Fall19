import os, torch, random
from model import Net_Fixed
from train_utils import Trainer
from lib import load_data, Data2Torch
from config import *
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model saving path
    from datetime import date
    date = date.today()
    out_model_fn = './model/%d%d%d/%s/' % (date.year, date.month, date.day, model_name)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    # load training and validation data (function inside lib.py)
    matrix_path = '../../../data_share/FBA/fall19/data/matrix/'
    trPC, vaPC = load_data(matrix_path)
    print(len(trPC))
    # prepare dataloader (function inside lib.py)
    t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
    v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), worker_init_fn=np.random.seed(manualSeed), **t_kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), worker_init_fn=np.random.seed(manualSeed), **v_kwargs)
    
    # build model (function inside model.py)
    model = Net_Fixed()
    model.to(device)
    
    # start training (function inside train_utils.py)
    Trer = Trainer(model, lr, epoch, out_model_fn)
    Trer.fit(tr_loader, va_loader, device)

    print(model_name)

if __name__ == "__main__":

    main()
    print(manualSeed)