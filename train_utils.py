from lib import distance_loss, classify_loss
import torch.optim as optim
import time
from torch.autograd import Variable

class Trainer:
    def __init__(self, PCmodel, SCmodel, lr, epoch, save_fn):
        self.epoch = epoch
        self.PCmodel = PCmodel
        self.SCmodel = SCmodel
        self.lr = lr
        self.save_fn = save_fn

        print('Start Training #Epoch:%d'%(epoch))

    def fit(self, tr_loader, va_loader, we):
        st = time.time()

        #define object
        save_dict = {}
        save_dict['tr_loss'] = []
        best_loss = 1000000000
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        for e in range(1, self.epoch+1):
            #learning rate
            lr = self.lr / (((e//(70*1))*2)+1) 
            loss_total = 0
            self.model.train()
            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))
            
            # Training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()  

                pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),
                
                #predict latent vector 
                pitch_v = self.PCmodel(pitch)
                score_v = self.SCmodel(score)
                
                #calculate loss
                loss_train = distance_loss(pitch_v, score_v, target)  
                loss_train.backward()
                opt.step()

            # Validate
            for batch_idx, _input in enumerate(va_loader):
                pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),
                
                #predict latent vector 
                pitch_v = self.PCmodel(pitch)
                score_v = self.SCmodel(score)
                
                #calculate loss
                loss_val = distance_loss(pitch_v, score_v, target) 
                    
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d] Loss_train %4f  Loss_val %4f  Time %d'
                    %(e, self.epoch, batch_idx+1, len(tr_loader), loss_train, loss_val, time.time() - st))
            sys.stdout.flush()
            print ('\n')

            if loss_val < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn+'e_%d'%(e))