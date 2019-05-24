import data_ops as ld
import sklearn.model_selection as ms
import joblib
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DTC
import datetime
import numpy as np
import os

train_features, train_labels = ld.load_train()
train_f, test_f, train_l, test_l = ld.split(train_features, train_labels, 0.2)

test_inputs = ld.load_test()
N_CLASS = train_labels.shape[1]
FEATURE_SIZE = train_features.shape[1]

train_l = np.argmax(train_l, axis=1)
test_l = np.argmax(test_l, axis=1)
max_tr, max_test = 0.0, 0.0
best_m = None
for i in range(5000):
    model = DTC(splitter='random', min_samples_split=10)
    model.fit(train_f, train_l)
    train_score, test_score = model.score(train_f, train_l), model.score(test_f, test_l)
    if test_score > max_test:
        max_tr = train_score
        max_test = test_score
        best_m = model
    if (i + 49) % 50 == 0:
        print(max_tr, max_test, i)

if max_test >= 0.85:
    preds = best_m.predict(test_inputs)
    ld.save_predictions(preds, os.getcwd(), 0)
    joblib.dump(best_m, 'dct-' + datetime.datetime.now().strftime('%Y%m%d-%H%M') + '-' + str(i) + '.joblib')
