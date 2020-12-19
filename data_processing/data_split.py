# Generate train_test split for spectral dataset
import numpy as np
import dill

# specify the band
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

PATH_FBA_DILL = "/media/SSD/FBA/saved_dill/"
PATH_FBA_MIDI = "/media/SSD/FBA/fall19/data/midi/"
PATH_FBA_MTX = "/media/SSD/FBA/fall19/data/matrix/"
PATH_FBA_SPLIT = "/media/SSD/FBA/split_dill/"
cpu_num = 3

def generate_newdata_newsplit(band):
    '''
    Generate data split train:valid:test = 8:1:1
    :param band: 'middle' or 'symphonic'
    '''
    dill_name = {'middle': 'middle_2_pc_6.dill', 'symphonic': 'symphonic_2_pc_6.dill'}
    total_name = {'middle':2611, 'symphonic': 2994}

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

    train_data = perf_data[ind[0:num_train]]
    valid_data = perf_data[ind[num_train:num_train + num_valid]]
    test_data = perf_data[ind[num_train + num_valid:num_train + 2 * num_valid]]

    with open('{}{}_2_pc_{}_train.dill'.format(PATH_FBA_SPLIT, band, len(YEAR)), 'wb') as f:
        dill.dump(train_data, f)
    with open('{}{}_2_pc_{}_test.dill'.format(PATH_FBA_SPLIT, band, len(YEAR)), 'wb') as f:
        dill.dump(test_data, f)
    with open('{}{}_2_pc_{}_valid.dill'.format(PATH_FBA_SPLIT, band, len(YEAR)), 'wb') as f:
        dill.dump(valid_data, f)

def generate_instrument_split(band, split_set='test'):
    '''
    Generate data split for each instrument
    Experiments with instrument splitting are not reported in the paper

    :param band: 'middle' or 'symphonic'
    :param split_set: 'train', 'valid' or 'test'
    :return:
    '''

    data_dill = '{}{}_2_pc_{}_{}.dill'.format(PATH_FBA_SPLIT, band, 6, split_set)
    perf_data = dill.load(open(data_dill, 'rb'))
    perf_data = np.array(perf_data)

    perf_inst_data = {}

    for inst in INSTRUMENT:
        perf_inst_data[inst] = []

    for perf in perf_data:
        inst = perf['instrumemt']
        perf_inst_data[inst].append(perf)

    for inst in INSTRUMENT:
        save_dill = '{}{}_2_pc_{}_{}_{}.dill'.format(PATH_FBA_SPLIT, band, 6, split_set, inst)
        print(save_dill, len(perf_inst_data[inst]))
        with open(save_dill, 'wb') as f:
            dill.dump(perf_inst_data[inst], f)

generate_newdata_newsplit('middle')
generate_newdata_newsplit('symphonic')

# generate_instrument_split('middle')
# generate_instrument_split('symphonic')