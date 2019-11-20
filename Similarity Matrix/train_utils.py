from lib import loss_func
import torch.optim as optim
import time, sys, torch
from torch.autograd import Variable
from tensorboard_logger import configure, log_value

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
        # e.g. model_name
        configure('runs/' + save_fn.split('/')[3], flush_secs=2)

    def fit(self, tr_loader, va_loader, device):
        st = time.time()

        #define object
        save_dict = {}
        save_dict['tr_loss'] = []
        best_loss = 1000000000

        for e in range(1, self.epoch+1):
            #learning rate (learning rate decay during training process)
            lr = self.lr / (((e//(70*1))*2)+1) 
            loss_total = 0
            self.model.train()
            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))

            # optimizer
            opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            #self.model.zero_grad()

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


            # Validate
            loss_val = 0
            for batch_idx, _input in enumerate(va_loader):
                matrix, target = Variable(_input[0].to(device)), Variable(_input[1].to(device))
                
                #predict latent vectors 
                pred = self.model(matrix.unsqueeze(1))
                
                #calculate loss
                loss_val += loss_func(pred, target).item()
                    
            # print model result
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Loss_train %4f  Loss_val %4f  Time %d'
                    %(e, self.epoch, loss_train, loss_val, time.time() - st))
            sys.stdout.flush()
            print ('\n')

            # log data for visualization later
            log_value('train_loss', loss_train, e)
            log_value('val_loss', loss_val, e)

            # save model
            if loss_val < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn+'model')
                best_epoch = e
                best_loss = loss_val

            # early stopping
            if (e-best_epoch) > 100:
                print(e, best_epoch)
                print('early stopping')
                break