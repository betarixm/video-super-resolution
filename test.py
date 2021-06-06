import os
import glob

import tensorflow as tf
import numpy as np
from PIL import Image

from nets import OurModel
from dataset import load_data
from train import lr_schedule, HUBER_DELTA

PATH = "FR_16_4.1622873040.034-0.00526"
TARGET_DIR = 0

if __name__ == "__main__":
    (x_train, y_train), (_, __) = load_data(num_dir=TARGET_DIR, train_ratio=1.0)

    model = OurModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA)
    )
    model.load_weights(PATH)

    model.evaluate(x_train, y_train)
    result = model.predict(x_train)
    path = "."

    dir_names_x = glob.glob(f"./input/LR/{TARGET_DIR:03d}")
    dir_inputs_x = [glob.glob(f"{d}/*") for d in dir_names_x]

    dir_counter = 0
    for dir_index, value in enumerate(dir_inputs_x):
        path_to_dir = os.path.join(path, "result", PATH, str(dir_index))
        os.makedirs(path_to_dir)
        for file_index in range(len(value) - 6):
            path_to_save = os.path.join(path_to_dir, str(file_index+3) + ".png")
            img = Image.fromarray(np.around(result[dir_counter][0] * 255).astype(np.uint8))
            if os.path.exists(path_to_save):
                os.remove(path_to_save)
            img.save(path_to_save)
            dir_counter += 1
