import data_ops as ld
import joblib
from sklearn.tree import DecisionTreeClassifier
import datetime
import numpy as np

train_features, train_labels = ld.load_train()
test_inputs = ld.load_test()
train_labels = np.argmax(train_labels, axis=1)
base = joblib.load('20190524-1730-acc70.model')
base_preds = base.predict(test_inputs)
base_acc = 0.7
max_iters = 100000
min_diff = base_preds.shape[0]
for i in range(max_iters):
    model = DecisionTreeClassifier(splitter='random', min_samples_split=12, max_depth=8)
    model.fit(train_features, train_labels)
    preds = model.predict(test_inputs)
    diff = np.size(np.where(preds != base_preds)[0])
    print(min_diff, '%' + str((i * 100) / max_iters))
    if diff < min_diff:
        min_diff = diff
    if diff < int((1 - base_acc) * base_preds.shape[0]):
        joblib.dump(model, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.model')
