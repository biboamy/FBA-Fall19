import datetime
import os
import torch.optim as optim
import sys
import time
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from lib import *
from model import *
from eval import evaluate_classification, evaluate_model


class Trainer:
    def __init__(self, model, lr, epoch, save_fn, beta=0.1, input_type='w_score', log=True):
        """
        Class for implementing an trainer
        """
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.beta = beta
        self.input_type = input_type
        self.writer = None
        self.log = log
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

        # configure tensorboardX summary writer
        current = datetime.datetime.now()
        if self.log:
            self.writer = SummaryWriter(
                logdir=os.path.join('runs/' + save_fn.split('/')[-2] + '_' + current.strftime("%m:%d:%H:%M"))
            )

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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

            self.update_lr(lr)

            # Training
            loss_train_recon = 0
            loss_train_score = 0
            loss_train_kld = 0
            self.model.train()
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()  

                # prepare inputs
                pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda())
                pitch = normalize_pitch(
                    pitch.view(pitch.shape[0] * pitch.shape[1], -1).unsqueeze(1)
                )
                score = normalize_midi(
                    score.view(score.shape[0] * score.shape[1], -1).unsqueeze(1)
                )
                target = target.view(target.shape[0] * target.shape[1], -1)

                if type(self.model) == PCPerformanceVAE:
                    # predict latent vectors
                    vae_out = self.model(pitch)
                    # calculate reconstruction loss
                    loss_reconstruct = mse_loss(vae_out[0].squeeze(), score.reshape(-1, score.shape[-1]))
                    # calculate performance loss
                    loss_score = mse_loss(vae_out[1], target)
                    # calculate latent loss
                    loss_kld = 0
                elif type(self.model) == PCPerformanceEncoder:
                    input_tensor = pitch
                    if self.input_type == 'w_score':
                        input_tensor = torch.cat((input_tensor, score), 1)
                    asses_score = self.model(input_tensor)
                    loss_score = mse_loss(asses_score, target)
                    loss_reconstruct = 0
                    loss_kld = 0
                elif type(self.model) == PCConvNet:
                    input_tensor = pitch
                    asses_score = self.model(input_tensor)
                    loss_score = mse_loss(asses_score, target)
                    loss_reconstruct = 0
                    loss_kld = 0
                else:
                    raise ValueError('invalid model type')

                # add losses and optimize
                total_loss = loss_reconstruct + self.beta * loss_score + loss_kld
                total_loss.backward()
                self.optimizer.step()
                loss_train_recon += loss_reconstruct
                loss_train_score += loss_score
                loss_train_kld += loss_kld

            # Validate
            loss_valid_recon = 0
            loss_valid_score = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, _input in enumerate(va_loader):
                    # prepare inputs
                    pitch, score, target = Variable(_input[0].cuda()), Variable(_input[1].cuda()), Variable(_input[2].cuda()),
                    pitch = normalize_pitch(
                        pitch.view(pitch.shape[0] * pitch.shape[1], -1).unsqueeze(1)
                    )
                    score = normalize_midi(
                        score.view(score.shape[0] * score.shape[1], -1).unsqueeze(1)
                    )
                    target = target.view(target.shape[0] * target.shape[1], -1)

                    if type(self.model) == PCPerformanceVAE:
                        # predict latent vectors
                        vae_out = self.model(pitch)
                        # calculate loss
                        loss_reconstruct = mse_loss(vae_out[0].squeeze(), score.reshape(-1, score.shape[-1]))
                        loss_score = mse_loss(vae_out[1], target)
                    elif type(self.model) == PCPerformanceEncoder:
                        input_tensor = pitch
                        if self.input_type == 'w_score':
                            input_tensor = torch.cat((input_tensor, score), 1)
                        asses_score = self.model(input_tensor)
                        loss_score = mse_loss(asses_score, target)
                        loss_reconstruct = 0
                    elif type(self.model) == PCConvNet:
                        input_tensor = pitch
                        asses_score = self.model(input_tensor)
                        loss_score = mse_loss(asses_score, target)
                        loss_reconstruct = 0
                    else:
                        raise ValueError('invalid model type')
                    loss_valid_recon += loss_reconstruct
                    loss_valid_score += loss_score

                # evaluate performance
                rsq, acc, corr, pval = evaluate_model(self.model, va_loader, input_type=self.input_type)

            # print model result
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Train Recons %2.3f  Train Score %2.3f  Train KL %2.3f  Valid Recons %2.3f  Valid Score %2.3f  Time %d'
                    %(e, self.epoch, loss_train_recon/len(tr_loader), loss_train_score/len(tr_loader), loss_train_kld/len(tr_loader), loss_valid_recon/len(va_loader), loss_valid_score/len(va_loader), time.time() - st))
            sys.stdout.flush()

            # log data for visualization later
            # log_value('train_loss', loss_train, e)
            # log_value('val_loss', loss_val, e)

            # log value in tensorboardX for visualization
            if self.log:
                self.writer.add_scalar('loss/train_recons', loss_train_recon/len(tr_loader), e)
                self.writer.add_scalar('loss/train_score', loss_train_score/len(tr_loader), e)
                self.writer.add_scalar('loss/train_kld', loss_train_kld/len(tr_loader), e)
                self.writer.add_scalar('loss/valid_recons', loss_valid_recon/len(va_loader), e)
                self.writer.add_scalar('loss/valid_score', loss_valid_score/len(va_loader), e)
                self.writer.add_scalar('metrics/r_squared', rsq, e)
                self.writer.add_scalar('metrics/accuracy', acc, e)
                # self.writer.add_scalar('metrics/correlation', corr, e)
                # self.writer.add_scalar('metrics/p_value', pval, e)
                self.writer.add_scalar('lr', lr, e)

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