from tensorflow import keras
from keras.preprocessing.image import array_to_img
import numpy as np
from PIL import Image
import glob
import os

from utils import LoadImage, depth_to_space_3D
from nets import FR_16, DynFilter

if __name__ == "__main__":
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

    model = keras.models.load_model(
        "./FR_16_4",
        custom_objects={
            "FR_16": FR_16,
            "DynFilter": DynFilter,
            "depth_to_space_3D": depth_to_space_3D
        }
    )
    model.evaluate(X_train, y_train)
    result = model.predict(X_train)

    i = 0
    path = "."
    for dir_index in range(len(dir_inputs_X)):
        path_to_dir = os.path.join(path, "result", str(dir_index))
        os.mkdir(path_to_dir)
        for file_index in range(len(dir) - 6):
            path_to_save = os.path.join(path_to_dir, str(file_index+3) + ".png")
            #img = array_to_img(result[i][0])
            #img = Image.fromarray(np.uint8(np.around(result[i][0])))
            img = Image.fromarray(np.around(result[i][0]*255).astype(np.uint8))
            img.save(path_to_save)
            i += 1