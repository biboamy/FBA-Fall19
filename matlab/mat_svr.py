from sklearn.svm import NuSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
from scipy.stats import pearsonr
import dill

FEAT = ["std", "nonscore", "score"]
segment = 2
band = "middle"
score = 0

def evaluate_classification(targets, predictions):
    print(targets.max(),targets.min(),predictions.max(),predictions.min(), len(predictions))
    #predictions[predictions>1]=1
    #predictions[predictions<0]=0
    #print(np.squeeze(targets), predictions)
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(np.squeeze(targets), np.squeeze(predictions))
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)

    return np.round(r2, decimals=3), np.round(accuracy, decimals=3), np.round(corrcoef, decimals=3), np.round(p, decimals=3)

for feat in FEAT:
    print("testing feature type: ", feat)

    train_data = dill.load(open('../../../data_share/FBA/fall19/data/mat/{}_{}_{}_{}_{}.dill'.format(band, segment, feat, 3, "train"), 'rb'))
    val_data = dill.load(open('../../../data_share/FBA/fall19/data/mat/{}_{}_{}_{}_{}.dill'.format(band, segment, feat, 3, "valid"), 'rb'))
    test_data = dill.load(open('../../../data_share/FBA/fall19/data/mat/{}_{}_{}_{}_{}.dill'.format(band, segment, feat, 3, "test"), 'rb'))

    scaler = MinMaxScaler()
    scaler.fit(train_data[0])

    print(scaler.data_max_.shape)

    train_label = train_data[1][:,score]
    val_label = val_data[1][:,score]
    test_label = test_data[1][:,score]

    train_data = scaler.transform(train_data[0])
    val_data = scaler.transform(val_data[0])
    test_data = scaler.transform(test_data[0])

    # train the nuSVR
    clf = NuSVR(C=1.0, kernel='linear', nu=0.5, gamma='auto')
    clf.fit(train_data, train_label)

    # predict on train set
    pred_train = clf.predict(train_data)
    print("train set metrics")
    print(evaluate_classification(train_label, pred_train))

    # predict on valid set
    pred_valid = clf.predict(val_data)
    print("valid set metrics")
    print(evaluate_classification(val_label, pred_valid))

    # predict on test set
    pred_test = clf.predict(test_data)
    print("test set metrics")
    print(evaluate_classification(test_label, pred_test))





