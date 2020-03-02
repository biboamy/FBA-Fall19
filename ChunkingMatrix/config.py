import random

# training parameters
batch_size = 32
num_workers = 1 # fixed
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.01
matrix_dim = 900 # resize dim
sub_num = 5
sub_matrix_dim = 300
score_choose = 0 #0: musicality, 1: note acc, 2: rhythmic acc, 3: tone quality
model_choose = 'ConvNet_Residual_LSTM_BatchNorm_Chunk_score'+str(score_choose) #'ConvNet_Fixed'
band = 'middle'

manualSeed = 10 #random.randint(0, 1000) #11

model_name = '{}_{}_batch{}_lr{}_{}'.format(model_choose, matrix_dim, batch_size, lr, manualSeed)
# model_name = '{}_{}_batch{}_lr{}'.format(model_choose, matrix_dim, batch_size, lr)
# model_name = 'ConvNet_Residual_LSTM_BatchNorm_Chunk_score0_300_batch32_lr0.01'

from datetime import date
date = date.today()
model_name_e = '%d%d%d/%s' % (date.year, date.month, date.day, model_name)
#model_name_e = '202031/%s' % (model_name)