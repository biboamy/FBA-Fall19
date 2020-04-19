import dill
import numpy as np
import os

if os.uname()[1] == 'mig1':
    PATH_FBA_DILL = "/media/SSD/FBA/saved_dill/"
    PATH_FBA_MIDI = "/media/SSD/FBA/fall19/data/midi/"
    PATH_FBA_MTX = "/media/SSD/FBA/fall19/data/matrix/"
    cpu_num = 3
else:
    PATH_FBA_DILL = "/media/Data/saved_dill/"
    PATH_FBA_MIDI = "/media/Data/fall19/data/midi/"
    PATH_FBA_MTX = "/media/Data/fall19/data/matrix/"
    cpu_num = 5

BAND = ['middle']
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

def check_num_by_year(dillFile):

    performances = dill.load(open(dillFile, 'rb'))

    cnt = {'2013':0, '2014':0, '2015':0, '2016':0, '2017':0, '2018':0}

    for perf in performances:
        cnt[perf['year']] = cnt[perf['year']] + 1
        #print(perf['year'], perf['pitch_contour'].shape)

    print(cnt)

def check_failed_perfs_and_remove(band):
    from data_matrix import load_data, computeDistanceMatrixAndAlignment

    if band == 'middle':
        fail_list = [('2016', '62406'), ('2016', '63386'), ('2016', '67310'), ('2016', '63920'), ('2016', '69847')]
        fail_idxs = []
        # ('2016', '62406'): pitch contour of length 0
        # ('2016', '63386'): all zeros in pitch contour
        # ('2016', '67310'): all zeros in pitch contour
        # ('2016', '63920'): all zeros in pitch contour
        # ('2016', '69847'): all zeros in pitch contour

        PC, SC = load_data(band='middle', midi_op='res12')

        print(len(PC))

        for i in np.arange(len(PC)):
            perf = PC[i]
            # print(perf['year'], perf['student_id'])
            if (perf['year'], str(perf['student_id'])) in fail_list:
                # t = computeDistanceMatrixAndAlignment(perf, SC)
                fail_idxs.append(i)

        fail_idxs = np.array(fail_idxs).astype(int)
        mask = np.ones(len(PC), dtype=bool)
        mask[fail_idxs] = False
        PC = PC[mask]

        print(len(PC))

        with open(PATH_FBA_DILL + 'middle_2_pc_6_fix.dill', 'wb') as f:
            dill.dump(list(PC), f)

    if band == 'symphonic':
        fail_list = [('2017', 78947), ('2017', 78952), ('2017', 79009), ('2017', 79013), ('2017', 79029), ('2017', 79061), ('2017', 79062), ('2017', 79068), ('2017', 79069), ('2017', 79094), ('2017', 79220), ('2017', 79221), ('2017', 79249), ('2017', 79277), ('2017', 79280), ('2017', 79284), ('2017', 79285), ('2017', 79287), ('2017', 79316), ('2017', 79318), ('2017', 79320), ('2017', 79321), ('2017', 79376), ('2017', 79451), ('2017', 79453), ('2017', 79820), ('2017', 79823), ('2017', 79887), ('2017', 80008), ('2017', 80309), ('2017', 80340), ('2017', 80358), ('2017', 80361), ('2017', 80418), ('2017', 80451), ('2017', 80460), ('2017', 80461), ('2017', 80462), ('2017', 80481)]

        fail_idxs = []

        PC, SC = load_data(band='symphonic', midi_op='res12')

        print(len(PC))

        for i in np.arange(len(PC)):
            perf = PC[i]
            # print(perf['year'], perf['student_id'])
            if (perf['year'], perf['student_id']) in fail_list:
                # t = computeDistanceMatrixAndAlignment(perf, SC)
                fail_idxs.append(i)

        fail_idxs = np.array(fail_idxs).astype(int)
        mask = np.ones(len(PC), dtype=bool)
        mask[fail_idxs] = False
        PC = PC[mask]

        print(len(PC))

        with open(PATH_FBA_DILL + 'symphonic_2_pc_6_fix.dill', 'wb') as f:
            dill.dump(list(PC), f)

# check_failed_perfs_and_remove('symphonic')

# dillFile = '/media/Data/saved_dill/symphonic_2_pc_6_fix.dill'
# check_num_by_year(dillFile)