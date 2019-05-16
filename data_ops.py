import csv
import numpy as np
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


def distinct_cols(matrix: np.ndarray):
    matrix = matrix.copy()
    row_size = matrix.shape[0]
    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        for c_i in range(col + 1, matrix.shape[1]):
            column_i = matrix[:, c_i]
            eq = np.equal(column, column_i)
            diff = eq[eq == True].shape[0]
            if diff == row_size:
                matrix[:, c_i] = -1
    return np.unique(np.where(matrix != -1)[1])
