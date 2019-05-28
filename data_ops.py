import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os

train_file = 'train.csv'
test_file = 'test.csv'


def load_train():
    inputs = []
    labels = []
    with open(train_file, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            try:
                row_inputs = row[:-1]
                label_i = int(row[-1])
                row_inputs = [float(i) for i in row_inputs]
                row_label = [0.0, 0.0]
                row_label[label_i] = 1.0
                inputs.append(row_inputs)
                labels.append(row_label)
            except:
                continue
    return np.asarray(inputs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def load_test():
    inputs = []
    with open(test_file, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            try:
                row_inputs = row
                row_inputs = [float(i) for i in row_inputs]
                inputs.append(row_inputs)
            except:
                continue
    return np.asarray(inputs, dtype=np.float32)


def save_predictions(predictions):
    with open(os.path.join(os.getcwd(), 'Submission-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'), 'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Predicted'])
        for i in range(predictions.shape[0]):
            writer.writerow([i + 1, predictions[i]])
