import tensorflow as tf
import numpy as np
import glob

from utils import LoadImage
from nets import OurModel

checkpoint_path = "checkpoint/FR_16_4.{epoch:03d}-{val_loss:.2f}"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    verbose=1
)


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


def train_and_evaluate():
    strategy = tf.distribute.MirroredStrategy()
    (X_train, y_train), (X_valid, y_valid) = load_data()
    with strategy.scope():
        model = OurModel()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(delta=0.01))
        history = model.fit(
            X_train,
            y_train,
            batch_size=16,
            epochs=128,
            validation_data=(X_valid, y_valid),
            callbacks=[checkpoint_callback]
        )
        model.save("./FR_16_4", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()



