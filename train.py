import data_ops as ld
import joblib
from sklearn.tree import DecisionTreeClassifier
import datetime
import numpy as np

(train_features, train_labels), test_features = ld.load_data('train.csv', 'test.csv')
base = joblib.load('20190524-1730-acc70.model')
base_preds = base.predict(test_features)
base_acc = 0.7
max_iters = 100000
min_diff = base_preds.shape[0]
for i in range(max_iters):
	model = DecisionTreeClassifier(splitter='random', min_samples_split=12, max_depth=8)
	model.fit(train_features, train_labels)
	test_preds = model.predict(test_features)
	diff = np.size(np.where(test_preds != base_preds)[0])
	if i % (max_iters // 100) == 0:
		print("Progress: %" + str((i * 100) / max_iters))
	if diff < min_diff:
		min_diff = diff
	if diff < int((1 - base_acc) * base_preds.shape[0]):
		print("Saving a model which has ", diff, " different predictions than the base model")
		joblib.dump(model, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.model')
print("Minimum different predictions from the base model: ", min_diff)