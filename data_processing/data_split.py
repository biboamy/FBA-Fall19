# Generate train_test split for spectral dataset
import numpy as np
import dill
import os

# specify the band
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

if os.uname()[1] == 'mig1':
    PATH_FBA_DILL = "/media/SSD/FBA/saved_dill/"
    PATH_FBA_MIDI = "/media/SSD/FBA/fall19/data/midi/"
    PATH_FBA_MTX = "/media/SSD/FBA/fall19/data/matrix/"
    PATH_FBA_SPLIT = "/media/SSD/FBA/split_dill/"
    cpu_num = 3
else:
    PATH_FBA_DILL = "/media/Data/saved_dill/"
    PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
    PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
    PATH_FBA_SPLIT = "/media/Data/split_dill/"
    cpu_num = 5

def generate_newdata_newsplit(band):
    dill_name = {'middle': 'middle_2_pc_6_fix.dill', 'symphonic': 'symphonic_2_pc_6_fix.dill'}
    total_name = {'middle':2611, 'symphonic': 2997}

    np.random.seed(1)

    data_path = PATH_FBA_DILL + dill_name[band]

    perf_data = dill.load(open(data_path, 'rb'))
    perf_data = np.array(perf_data)
    print(len(perf_data))
    ind = np.arange(total_name[band])
    np.random.shuffle(ind)

    total = len(ind)
    num_valid = int(total * 0.1)
    num_train = int(0.8 * total)

    # for i in range(num_train):
    #     print(perf_data[ind[i]]['student_id'])

    train_data = perf_data[ind[0:num_train]]
    valid_data = perf_data[ind[num_train:num_train + num_valid]]
    test_data = perf_data[ind[num_train + num_valid:num_train + 2 * num_valid]]

    with open('{}{}_2_pc_{}_train.dill'.format(PATH_FBA_SPLIT, band, len(YEAR)), 'wb') as f:
        dill.dump(train_data, f)
    with open('{}{}_2_pc_{}_test.dill'.format(PATH_FBA_SPLIT, band, len(YEAR)), 'wb') as f:
        dill.dump(test_data, f)
    with open('{}{}_2_pc_{}_valid.dill'.format(PATH_FBA_SPLIT, band, len(YEAR)), 'wb') as f:
        dill.dump(valid_data, f)

def generate_newdata_oldsplit(band):
    PATH_FBA_DILL_OLD = '/media/Data/fall19/data/pitch_contour/'
    newdill_name = PATH_FBA_DILL + '{}_2_pc_6_fix.dill'.format(band)
    oldsplit_dill_name = {'train': '{}_2_pc_3_train.dill', 'valid': '{}_2_pc_3_valid.dill', 'test': '{}_2_pc_3_test.dill'}

    np.random.seed(1)

    perf_data_train_old = dill.load(open(PATH_FBA_DILL_OLD + oldsplit_dill_name['train'].format(band), 'rb'))
    print(len(perf_data_train_old))
    perf_data_train_yearid = [(perf['year'], perf['student_id']) for perf in perf_data_train_old]
    del perf_data_train_old

    perf_data_valid_old = dill.load(open(PATH_FBA_DILL_OLD + oldsplit_dill_name['valid'].format(band), 'rb'))
    print(len(perf_data_valid_old))
    perf_data_valid_yearid = [(perf['year'], perf['student_id']) for perf in perf_data_valid_old]
    del perf_data_valid_old

    perf_data_test_old = dill.load(open(PATH_FBA_DILL_OLD + oldsplit_dill_name['test'].format(band), 'rb'))
    print(len(perf_data_test_old))
    perf_data_test_yearid = [(perf['year'], perf['student_id']) for perf in perf_data_test_old]
    del perf_data_test_old

    perf_data_all_new = dill.load(open(newdill_name, 'rb'))
    perf_data_all_new = np.array(perf_data_all_new)

    train_idx = []
    valid_idx = []
    test_idx = []

    for i in np.arange(len(perf_data_all_new)):
        perf = perf_data_all_new[i]
        if (perf['year'], perf['student_id']) in perf_data_train_yearid:
            train_idx.append(i)
        elif (perf['year'], perf['student_id']) in perf_data_valid_yearid:
            valid_idx.append(i)
        elif (perf['year'], perf['student_id']) in perf_data_test_yearid:
            test_idx.append(i)
        else:
            pass
    print(len(train_idx), len(valid_idx), len(test_idx))

    train_data = perf_data_all_new[np.array(train_idx).astype(int)]
    valid_data = perf_data_all_new[np.array(valid_idx).astype(int)]
    test_data = perf_data_all_new[np.array(test_idx).astype(int)]

    with open('{}{}_2_pc_{}_train_oldsplit.dill'.format(PATH_FBA_SPLIT, band, 3), 'wb') as f:
        dill.dump(train_data, f)
    with open('{}{}_2_pc_{}_test_oldsplit.dill'.format(PATH_FBA_SPLIT, band, 3), 'wb') as f:
        dill.dump(test_data, f)
    with open('{}{}_2_pc_{}_valid_oldsplit.dill'.format(PATH_FBA_SPLIT, band, 3), 'wb') as f:
        dill.dump(valid_data, f)

# generate_newdata_newsplit('symphonic')

# generate_newdata_oldsplit('symphonic')