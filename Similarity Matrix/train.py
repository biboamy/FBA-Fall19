import os, torch
from model import Net, Net_Fixed
from train_utils import Trainer
from lib import load_data, Data2Torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

# training parameters
batch_size = 16
num_workers = 1 # fixed
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.00001
model_choose = 'ConvNet_Fixed'

def main():

    model_name = '{}_batch{}_lr{}'.format(model_choose, batch_size, lr)

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
    
    # prepare dataloader (function inside lib.py)
    t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
    v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch([trPC]), **t_kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([vaPC]), **v_kwargs)
    
    # build model (function inside model.py)
    if model_choose == 'ConvNet':
        model = Net()
    elif model_choose == 'ConvNet_Fixed':
        model = Net_Fixed()
    model.to(device)
    
    # start training (function inside train_utils.py)
    Trer = Trainer(model, lr, epoch, out_model_fn)
    Trer.fit(tr_loader, va_loader, device)

    print(model_name)

if __name__ == "__main__":

    main()