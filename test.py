import os
import glob

import tensorflow as tf
import numpy as np
from PIL import Image

from nets import OurModel
from dataset import load_data
from train import lr_schedule, HUBER_DELTA

if __name__ == "__main__":
    (x_train, y_train), (_, __) = load_data(num_dir=0, train_ratio=1.0)

    model = OurModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA)
    )
    model.load_weights("checkpoint/FR_16_4.1622718916.115-0.00215")

    model.evaluate(x_train, y_train)
    result = model.predict(x_train)
    path = "."

    dir_names_x = glob.glob('./input/LR/0')
    dir_inputs_x = [glob.glob(f"{d}/*") for d in dir_names_x]

    dir_counter = 0
    for dir_index, value in enumerate(dir_inputs_x):
        path_to_dir = os.path.join(path, "result", str(dir_index))
        os.mkdir(path_to_dir)
        for file_index in range(len(value) - 6):
            path_to_save = os.path.join(path_to_dir, str(file_index+3) + ".png")
            img = Image.fromarray(np.around(result[dir_counter][0] * 255).astype(np.uint8))
            img.save(path_to_save)
            dir_counter += 1
