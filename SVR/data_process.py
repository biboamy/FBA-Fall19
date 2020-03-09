import h5py
import dill
import numpy as np
import scipy.io

FEAT = ["std", "nonscore", "score"]
segment = 2
band = "middle"
INST = ["Alto Saxophone", "Bb Clarinet", "Flute"]
YEAR = ["2013", "2014", "2015"]

feat_dim = {"std": 68, "nonscore": 24, "score": 22}

def combineData():
    for feat in FEAT:
        data_feat = {}
        for inst in INST:
            data_feat[inst] = {}
            for year in YEAR:
                data_cur = {}
                mat_path = "/home/data_share/FBA/fall19/data/mat/{}{}{}_{}_{}.mat".format(band, inst, segment, feat, year)
                mat = scipy.io.loadmat(mat_path)
                assert(mat['features'].shape[0] == mat['labels'].shape[0] and mat['labels'].shape[0] == mat['student_ids'].shape[0])
                for i in range(mat['student_ids'].shape[0]):
                    data_cur[mat['student_ids'][i,0]] = {'features': mat['features'][i, :], 'labels': mat['labels'][i, :]}
                data_feat[inst][year] = data_cur
        save_file = '/home/data_share/FBA/fall19/data/mat/{}_{}_{}_{}.dill'.format(band, segment, feat, len(YEAR))
        with open(save_file, 'wb') as f:
            dill.dump(data_feat, f)


def createMatrixDill(feat, PC, data_split):

    new_pc_file = '../../../data_share/FBA/fall19/data/mat/{}_{}_{}_{}.dill'.format(band, segment, feat, len(YEAR))
    save_dill_file = '../../../data_share/FBA/fall19/data/mat/{}_{}_{}_{}_{}.dill'.format(band, segment, feat, len(YEAR), data_split)

    mat_data = dill.load(open(new_pc_file, 'rb'))

    data_features = np.empty((0, feat_dim[feat]))
    data_labels = np.empty((0, 4))
    missing_ids = []
    for pc in PC:
        year = pc['year']
        student_id = pc['student_id']
        inst = pc['instrumemt']
        ratings = np.array(pc['ratings'])  # choose which score
        # new_ratings = mat_data[inst][year][student_id]['labels']
        try:
            # print(mat_data[inst][year][student_id]['features'].shape)
            data_features = np.concatenate((data_features, mat_data[inst][year][student_id]['features'].reshape(1, -1)), axis=0)
            data_labels = np.concatenate((data_labels, ratings.reshape(1, -1)), axis=0)
        except:
            missing_ids.append(student_id)

    print(missing_ids)
    with open(save_dill_file, 'wb') as f:
        dill.dump([data_features, data_labels], f)

#combineData()

orig_pc_file = '../../../data_share/FBA/fall19/data/pitch_contour/{}_2_pc_3_'.format(band)
# train
trPC = np.array(dill.load(open(orig_pc_file + 'train.dill', 'rb')))
# valid
vaPC = np.array(dill.load(open(orig_pc_file + 'valid.dill', 'rb')))
# test
tePC = np.array(dill.load(open(orig_pc_file + 'test.dill', 'rb')))

splits = ["train", "valid", "test"]

for feat in FEAT:
    createMatrixDill(feat, tePC, "test")
    createMatrixDill(feat, vaPC, "valid")
    createMatrixDill(feat, trPC, "train")