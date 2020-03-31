import datetime
import os
import torch.optim as optim
import sys
import time
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tensorboard_logger import configure, log_value
from PerformanceVAE.lib import *


class Trainer:
    def __init__(self, model, lr, epoch, save_fn, beta=0.1):
        """
        Class for implementing an trainer
        """
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.beta = beta
        self.writer = None

        print('Start Training #Epoch:%d'%(epoch))

        # configure tensorboardX summary writer
        current = datetime.datetime.now()
        self.writer = SummaryWriter(
            logdir=os.path.join('runs/' + save_fn.split('/')[-2] + '_' + current.strftime("%m:%d:%H:%M"))
        )

    def fit(self, tr_loader, va_loader):
        st = time.time()

        # define object
        save_dict = dict()
        save_dict['tr_loss'] = []
        best_loss = 1000000000

        for e in range(1, self.epoch+1):
            # learning rate (learning rate decay during training process)
            lr = self.lr / (((e//(70*1))*2)+1) 
            self.model.train()
            print('\n==> Training Epoch #%d lr=%4f'%(e, lr))

            # optimizer
            opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            
            # Training
            loss_train_recon = 0
            loss_train_score = 0
            loss_train_kld = 0
            self.model.train()
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()  

                pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),
                
                # predict latent vectors
                vae_out = self.model(pitch)
                # calculate reconstruction loss
                loss_reconstruct = mse_loss(vae_out[0].squeeze(), score.reshape(-1, score.shape[-1]))

                # calculate performance loss
                loss_score = mse_loss(vae_out[1].squeeze(), target.reshape(-1))

                # calculate latent loss
                dist_loss = compute_kld_loss(
                    vae_out[2], vae_out[3], beta=self.beta
                )
                # add losses and optimize
                total_loss = loss_reconstruct + loss_score + dist_loss
                total_loss.backward()
                opt.step()
                loss_train_recon += loss_reconstruct
                loss_train_score += loss_score
                loss_train_kld += dist_loss

            # Validate
            loss_valid_recon = 0
            loss_valid_score = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, _input in enumerate(va_loader):
                    pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),

                    #predict latent vectors
                    vae_out = self.model(pitch)

                    #calculate loss
                    loss_reconstruct = mse_loss(vae_out[0].squeeze(), score.reshape(-1, score.shape[-1]))
                    loss_score = mse_loss(vae_out[1].squeeze(), target.reshape(-1))
                    loss_valid_recon += loss_reconstruct
                    loss_valid_score += loss_score

            # print model result
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Train recon %2.3f  Train score %2.3f  Train KL %2.3f  Valid score %2.3f  Valid score %2.3f  Time %d'
                    %(e, self.epoch, loss_train_recon/len(tr_loader), loss_train_score/len(tr_loader), loss_train_kld/len(tr_loader), loss_valid_recon/len(va_loader), loss_valid_score/len(va_loader), time.time() - st))
            sys.stdout.flush()

            # log data for visualization later
            # log_value('train_loss', loss_train, e)
            # log_value('val_loss', loss_val, e)

            # log value in tensorboardX for visualization
            self.writer.add_scalar('loss/train_recons', loss_train_recon, e)
            self.writer.add_scalar('loss/train_score', loss_train_score, e)
            self.writer.add_scalar('loss/train_kld', loss_train_kld, e)
            self.writer.add_scalar('loss/valid_recons', loss_valid_recon, e)
            self.writer.add_scalar('loss/valid_score', loss_valid_score, e)

            # save model
            if (loss_valid_score) < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn+'model')
                best_epoch = e
                best_loss = loss_valid_score

            # early stopping
            if (e-best_epoch) > 100:
                print(e, best_epoch)
                print('early stopping')
                break