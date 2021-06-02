import pickle

import numpy as np
import glob

from utils import LoadImage


def load_data(train_ratio=0.75):
    x_train = []
    y_train = []

    dir_names_x = glob.glob('./input/LR/*')
    dir_names_y = glob.glob('./input/HR/*')

    dir_inputs_x = [glob.glob(f"{d}/*") for d in dir_names_x]
    dir_inputs_y = [glob.glob(f"{d}/*") for d in dir_names_y]

    for x, y in zip(dir_inputs_x, dir_inputs_y):
        x.sort(), y.sort()

    dir_files_x = [list([LoadImage(file) for file in d]) for d in dir_inputs_x]
    dir_files_y = [list([LoadImage(file) for file in d]) for d in dir_inputs_y]

    for d_x, d_y in zip(dir_files_x, dir_files_y):
        assert len(d_x) == len(d_y)

        for i in range(len(d_x) - 6):
            x_train.append(d_x[i:i + 7])
            y_train.append(d_y[i + 3])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_valid = x_train[int(len(x_train) * train_ratio):]
    y_valid = y_train[int(len(y_train) * train_ratio):]
    x_train = x_train[:int(len(x_train) * train_ratio)]
    y_train = y_train[:int(len(y_train) * train_ratio)]

    return (x_train, y_train), (x_valid, y_valid)


if __name__ == "__main__":
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(load_data(), f)
