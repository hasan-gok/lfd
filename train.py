import data_ops as ld
import sklearn.model_selection as ms
import joblib
from sklearn.tree import DecisionTreeClassifier
import datetime
import numpy as np
import os

train_inputs, train_labels = ld.load_train()
train_labels = np.argmax(train_labels, axis=1)
train_features, test_features, train_labels, test_labels = ms.train_test_split(train_inputs, train_labels,
                                                                               test_size=0.1,
                                                                               shuffle=True, random_state=35)
test_inputs = ld.load_test()
NROF_CLASSES = 2
FEATURE_SIZE = train_features.shape[1]

for i in range(500):
    dt = DecisionTreeClassifier(criterion='entropy', splitter='random')
    dt.fit(train_features, train_labels)
    acc = dt.score(test_features, test_labels)
    tr_acc = dt.score(train_features, train_labels)
    print(acc, tr_acc)
    if acc > 0.9 and tr_acc > 0.9:
        predictions = dt.predict(test_inputs)
        ld.save_predictions(predictions, os.getcwd(), i)
        joblib.dump(dt, 'dt-' + datetime.datetime.now().strftime('%Y%m%d-%H%M') + '-' + str(i) + '.joblib')
