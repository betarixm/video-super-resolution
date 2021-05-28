from tensorflow import keras
import numpy as np
from PIL import Image
import glob
import os

from utils import LoadImage

if __name__ == "__main__":
    X_train = []
    y_train = []

    dir_names_X = glob.glob('./input/LR/*')
    dir_inputs_X = []
    dir_inputs_y = []
    for dir in dir_names_X:
        dir_inputs_X.append(glob.glob(dir + '/*'))
    for dir in dir_inputs_X:
        dir.sort()

    dir_files_X = []
    for dir in dir_inputs_X:
        temp_dir_X = []
        for file in dir:
            temp_dir_X.append(LoadImage(file))
        dir_files_X.append(temp_dir_X)

    for dir in dir_files_X:
        for i in range(len(dir) - 6):
            X_train.append(dir[i:i + 7])

    X_train = np.asarray(X_train)

    model = keras.models.load_model("./FR_16(4)")
    result = model.predict(X_train)

    i = 0
    path = "."
    for dir_index in range(len(dir_inputs_X)):
        path_to_dir = os.path.join(path, "result", str(dir_index))
        os.mkdir(path_to_dir)
        for file_index in range(len(dir) - 6):
            path_to_save = os.path.join(path_to_dir, str(file_index+3) + ".png")
            Image.fromarray(np.uint8(np.around(result[i][0] * 255))).save(path_to_save)
            i += 1