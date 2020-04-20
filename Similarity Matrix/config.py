# path
PATH_FBA_DILL = "/media/Data/saved_dill/"
PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
PATH_FBA_SPLIT = "/media/Data/split_dill/"

band = "symphonic" # middle, symphonic
split = "new" # old: 2013~2015, new: 2013~2018

data_all_pc = "{}_2_pc_6_fix.dill"

if split == "old":
	data_train_mtx = "{}_matrix_fixed_train{}_oldsplit.dill"
	data_valid_mtx = "{}_matrix_fixed_valid{}_oldsplit.dill"
	data_test_mtx = "{}_matrix_fixed_test{}_oldsplit.dill"
elif split == "new":
	data_train_mtx = "{}_matrix_fixed_train{}.dill"
	data_valid_mtx = "{}_matrix_fixed_valid{}.dill"
	data_test_mtx = "{}_matrix_fixed_test{}.dill"

# training parameters
batch_size = 32
num_workers = 4 # fixed
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.05
isNorm = True
matrix_dim = 600 # resize dim
score_choose = 3 #0: musicality, 1: note acc, 2: rhythmic acc, 3: tone quality
model_choose = 'ConvNet_Residual_BatchNorm_score'+str(score_choose) #'ConvNet_Fixed'

manualSeed = 10 # random.randint(0, 1000) #11

#model_name = '{}_{}_batch{}_lr{}_{}_{}_isNorm{}'.format(model_choose, matrix_dim, batch_size, lr, band, split,isNorm)

#from datetime import date
#date = date.today()
#model_name_e = '%d%d%d/%s' % (date.year, date.month, date.day, model_name)
