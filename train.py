## -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob

from utils import LoadImage
from nets import OurModel

def load_data():
    X_train = []
    y_train = []

    dir_names_X = glob.glob('./input/LR/*')
    dir_names_y = glob.glob('./input/HR/*')
    dir_inputs_X = []
    dir_inputs_y = []
    for dir in dir_names_X:
        dir_inputs_X.append(glob.glob(dir + '/*'))
    for dir in dir_names_y:
        dir_inputs_y.append(glob.glob(dir + '/*'))
    for dir in dir_inputs_X:
        dir.sort()
    for dir in dir_inputs_y:
        dir.sort()

    dir_files_X = []
    dir_files_y = []
    for dir in dir_inputs_X:
        temp_dir_X = []
        for file in dir:
            temp_dir_X.append(LoadImage(file))
        dir_files_X.append(temp_dir_X)
    for dir in dir_inputs_y:
        temp_dir_y = []
        for file in dir:
            temp_dir_y.append(LoadImage(file))
        dir_files_y.append(temp_dir_y)

    for dir in dir_files_X:
        for i in range(len(dir) - 6):
            X_train.append(dir[i:i + 7])
    for dir in dir_files_y:
        for i in range(len(dir) - 6):
            y_train.append(dir[i + 3])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_valid = X_train[int(len(X_train) * 3 / 4):]
    y_valid = y_train[int(len(y_train) * 3 / 4):]
    X_train = X_train[:int(len(X_train) * 3 / 4)]
    y_train = y_train[:int(len(y_train) * 3 / 4)]

    return (X_train, y_train), (X_valid, y_valid)


def train_and_evaluate():

    (X_train, y_train), (X_valid, y_valid) = load_data()
    model = OurModel()
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.Huber(delta=0.01))
    history = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=22,
        validation_data=(X_valid, y_valid)
    )
    model.save("./FR_16_4", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()



