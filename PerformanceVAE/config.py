# path
PATH_FBA_DILL = "/media/Data/saved_dill/"
PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
PATH_FBA_SPLIT = "/media/Data/split_dill/"

band = "middle"
split = "new"  # old: 2013~2015, new: 2013~2018

data_all_pc = "{}_2_pc_6_fix.dill"

data_train_pc = {"old": "{}_2_pc_3_train_oldsplit.dill", "new": "{}_2_pc_6_train.dill"}
data_valid_pc = {"old": "{}_2_pc_3_valid_oldsplit.dill", "new": "{}_2_pc_6_valid.dill"}
data_test_pc = {"old": "{}_2_pc_3_test_oldsplit.dill", "new": "{}_2_pc_6_test.dill"}

midi_aligned_s = "{}_2_midi_aligned_s_6.dill"

feat = 'pitch contour'  # fixed
midi_op = 'aligned_s'  # fixed
model_choose = "PerformanceEncoder"  # fixed
score_choose = 0  # 0: musicality, 1: note acc., 2: rhythmic acc. 3: tone quality

# training parameters
batch_size = 64
num_workers = 2  # fixed
shuffle = True  # fixed
epoch = 1000  # fixed
lr = 1e-4
log = True

process_collate = 'windowChunk'  # 'randomChunk', 'windowChunk', 'padding'
sample_num = 1  # numbers of chunks # if choosing windowChunk, sample_num has to be 1
chunk_size = 2000  # 1000 ~ 5 sec / 2000 ~ 10 sec

# model parameters
dropout_prob = 0.5
num_rec_layers = 2
z_dim = 16
kernel_size = 7
stride = 1
num_conv_features = 4
input_type = 'no_score'  # 'w_score', 'no_score'

# normalization params
pitch_norm_coeff = 12543.854   # 4186
midi_norm_coeff = 127.0        # 108

# evaluation parameters
overlap_flag = False