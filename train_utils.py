from lib import distance_loss, classify_loss
import torch.optim as optim
import time, sys, torch
from torch.autograd import Variable

class Trainer:
    def __init__(self, model, lr, epoch, save_fn):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn

        print('Start Training #Epoch:%d'%(epoch))

    def fit(self, tr_loader, va_loader):
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
            
            # Training
            loss_train = 0
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()  

                pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),
                
                #predict latent vectors 
                pitch_v, score_v = self.model(pitch, score)
                
                #calculate loss
                loss = distance_loss(pitch_v, score_v, target) [0] 
                loss.backward()
                opt.step()
                loss_train += loss

            # Validate
            loss_val = 0
            for batch_idx, _input in enumerate(va_loader):
                pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),
                
                #predict latent vectors 
                pitch_v, score_v = self.model(pitch, score)
                
                #calculate loss
                loss_val += distance_loss(pitch_v, score_v, target) [0]
                    
            # print model result
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Loss_train %4f  Loss_val %4f  Time %d'
                    %(e, self.epoch, loss_train, loss_val, time.time() - st))
            sys.stdout.flush()
            print ('\n')

            # save model
            if loss_val < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn+'model')
                best_epoch = e
                best_loss = loss_val

            # early stopping
            print(e, best_epoch)
            if (e-best_epoch) > 50:
                print('early stopping')
                break