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


def save_predictions(predictions, path, i):
    if not os.path.isdir(path):
        raise NotADirectoryError
    with open(os.path.join(path, 'Submission-' + datetime.now().strftime('%Y%m%d-%H%M') + '-' + str(i) + '.csv'), 'w',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Predicted'])
        for i in range(predictions.shape[0]):
            writer.writerow([i + 1, predictions[i]])


def plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    train_loss, val_loss, train_acc, val_acc = history['loss'], history['val_loss'], history['binary_accuracy'], \
                                               history['val_binary_accuracy']
    ax1.plot(np.arange(0, len(train_loss), 1), history['loss'], label='Training loss')
    ax1.plot(np.arange(0, len(val_loss), 1), history['val_loss'], label='Validation loss')
    ax1.set_ylim((0, 10))
    ax2.plot(np.arange(0, len(train_acc), 1), history['binary_accuracy'], label='Training accuracy')
    ax2.plot(np.arange(0, len(val_acc), 1), history['val_binary_accuracy'], label='Validation accuracy')
    ax2.set_ylim((0, 1))
    plt.show()


def split(features, labels, test_size):
    n_samples = labels.shape[0]
    n_classes = labels.shape[1]
    if test_size > 1:
        test_size = test_size // n_samples
    train, test = None, None
    for i in range(n_classes):
        sample_i = np.where(labels[:, i] == 1)[0]
        np.random.shuffle(sample_i)
        t_size = int(sample_i.shape[0] * test_size)
        if i == 0:
            test = sample_i[:t_size]
            train = sample_i[t_size:]
        else:
            test = np.concatenate([test, sample_i[:t_size]])
            train = np.concatenate([train, sample_i[t_size:]])
    np.random.shuffle(train)
    np.random.shuffle(test)
    return features[train], features[test], labels[train], labels[test]
