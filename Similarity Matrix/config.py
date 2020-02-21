import random

# training parameters
batch_size = 32
num_workers = 1 # fixed
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.05
matrix_dim = 600 # resize dim
model_choose = 'ConvNet_Residual_BatchNorm' #'ConvNet_Fixed'
band = 'middle'

manualSeed = 10#random.randint(0, 1000) #11

model_name = '{}_{}_batch{}_lr{}_{}'.format(model_choose, matrix_dim, batch_size, lr, manualSeed)

from datetime import date
date = date.today()
model_name_e = '%d%d%d/%s' % (date.year, date.month, date.day, model_name)