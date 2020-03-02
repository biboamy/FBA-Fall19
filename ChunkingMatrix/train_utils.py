from lib import loss_func
import torch.optim as optim
import time, sys, torch
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from sklearn import metrics

class Trainer:
    def __init__(self, model, lr, epoch, save_fn):
        """
        Class for implementing an trainer
        """
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn

        print('Start Training #Epoch:%d'%(epoch))

        # declare save file name
        file_info = 'tensorlog'

        # configure tensor-board logger
        import datetime

        current = datetime.datetime.now()

        # configure tensor-board logger
        # configure('runs/' + save_fn.split('/')[-2], flush_secs=2)

    def fit(self, tr_loader, va_loader, device):
        st = time.time()

        #define object
        save_dict = {}
        save_dict['tr_loss'] = []
        best_loss = 1000000000

        opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.5, last_epoch=-1)

        for e in range(1, self.epoch+1):

            loss_total = 0
            self.model.train()
            if e % 70 == 0:
                scheduler.step()
            print('\n==> Training Epoch #%d' % (e))
            for param_group in opt.param_groups:
                print("lr: ", param_group['lr'])

            loss_train = 0
            # Training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()
                matrix, target = Variable(_input[0].to(device)), Variable(_input[1].to(device))

                #predict latent vectors 
                pred = self.model(matrix.unsqueeze(1))

                #calculate loss
                loss = loss_func(pred, target)
                loss.backward()
                opt.step()

                loss_train += loss.item()
                #print(loss_train)
                #input()

            loss_train = loss_train / len(tr_loader)

            # Validate
            loss_val = 0
            r2_val = 0
            self.model.eval()
            for batch_idx, _input in enumerate(va_loader):
                matrix, target = Variable(_input[0].to(device)), Variable(_input[1].to(device))

                for i in [1, 4, 7]:
                    self.model.model.conv[i].bn1.momentum = 0
                    self.model.model.conv[i].bn2.momentum = 0
                    self.model.model.conv[i].bn3.momentum = 0
                    self.model.model.conv[i].bn1.track_running_stats = False
                    self.model.model.conv[i].bn2.track_running_stats = False
                    self.model.model.conv[i].bn3.track_running_stats = False

                #predict latent vectors 
                pred = self.model(matrix.unsqueeze(1))
                #calculate loss
                loss_val += loss_func(pred, target).item()
                if pred.shape[0] > 1:
                    r2_val += metrics.r2_score(target.data.cpu().numpy(), pred.data.cpu().numpy())

            loss_val = loss_val  / len(va_loader)
            r2_val = r2_val / (len(va_loader) - (pred.shape[0] <= 1))

            # print model result
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Loss_train %4f  Loss_val %4f  R2_val %4f  Time %d'
                    %(e, self.epoch, loss_train, loss_val, r2_val, time.time() - st))
            sys.stdout.flush()
            #print ('\n')

            # log data for visualization later
            # log_value('train_loss', loss_train, e)
            # log_value('val_loss', loss_val, e)
            # log_value('val_r2', r2_val, e)

            # save model
            if loss_val < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn+'model')
                best_epoch = e
                best_loss = loss_val

            # early stopping
            if (e-best_epoch) > 20:
                print(e, best_epoch)
                print('early stopping')
                break