# path
PATH_FBA_DILL = "/media/Data/saved_dill/"
PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
PATH_FBA_SPLIT = "/media/Data/split_dill/"

band = "symphonic" # middle, symphonic
split = "old" # old: 2013~2015, new: 2013~2018

data_all_pc = "{}_2_pc_6_fix.dill"

data_train_pc = {"old": "{}_2_pc_3_train_oldsplit.dill", "new": "{}_2_pc_6_train.dill"}
data_valid_pc = {"old": "{}_2_pc_3_valid_oldsplit.dill", "new": "{}_2_pc_6_valid.dill"}
data_test_pc = {"old": "{}_2_pc_3_test_oldsplit.dill", "new": "{}_2_pc_6_test.dill"}

midi_aligned_s = "{}_2_midi_aligned_s_6.dill"

mtx_orig_h5 = "{}_2_6_matrix.h5"
id2idx_file = "{}_id2idx_6.dill"

# training parameters
batch_size = 32
num_workers = 1 # fixed: HDF5 does not allow multiprocessing
shuffle = True # fixed
epoch = 1000 # fixed
lr = 0.05

isNorm = True
chunk_matrix_dim = 400 # resize dim; the original submatrix should be: midi_snippet_len x (chunk_size/5)
score_choose = 0 # 0: musicality, 1: note acc, 2: rhythmic acc, 3: tone quality

process_collate = 'randomChunk' # 'randomChunk', 'windowChunk', 'padding'
sample_num = 2 # numbers of chunks # if choosing windowChunk, sample_num has to be 1
chunk_size = 2000 # 1000 ~ 5 sec / 2000 ~ 10 sec
overlap_flag = False

model_choose = 'ConvNet_Residual_BatchNorm_score'+str(score_choose) #'ConvNet_Fixed'

manualSeed = 10 # random.randint(0, 1000) #11

model_name = 'ChunkedInput_{}_{}_batch{}_lr{}_{}_{}_isNorm{}'.format(model_choose, chunk_matrix_dim, batch_size, lr, band, split,isNorm)

from datetime import date
date = date.today()
model_name_e = '%d%d%d/%s' % (date.year, date.month, date.day, model_name)
# model_name_e = '2020419/ConvNet_Residual_BatchNorm_score0_400_batch32_lr0.05_symphonic_old_isNormTrue'
