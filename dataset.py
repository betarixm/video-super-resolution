import pickle
import glob
import multiprocessing

from PIL import Image
import numpy as np


def load_image(path, color_mode='RGB', channel_mean=None, mod_crop=None):
    """
    Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    """

    if mod_crop is None:
        mod_crop = [0, 0, 0, 0]

    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')
    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:, :, 0:1]
    else:
        raise Exception(f"{color_mode} is not supported.")

    # To 0-1
    x *= 1.0 / 255.0

    if channel_mean:
        x[:, :, 0] -= channel_mean[0]
        x[:, :, 1] -= channel_mean[1]
        x[:, :, 2] -= channel_mean[2]

    if mod_crop[0] * mod_crop[1] * mod_crop[2] * mod_crop[3]:
        x = x[mod_crop[0]:-mod_crop[1], mod_crop[2]:-mod_crop[3], :]

    return x


def load_data(train_ratio=0.75):
    manager = multiprocessing.Manager()
    m_dir_files = manager.dict()
    pool = multiprocessing.Pool()

    def worker(path: str):
        m_dir_files[path] = load_image(path)

    x_train = []
    y_train = []

    dir_names_x = glob.glob('./input/LR/*')
    dir_names_y = glob.glob('./input/HR/*')

    dir_inputs_x = [glob.glob(f"{d}/*") for d in dir_names_x]
    dir_inputs_y = [glob.glob(f"{d}/*") for d in dir_names_y]

    for x, y in zip(dir_inputs_x, dir_inputs_y):
        x.sort(), y.sort()

    target_x = [str(file) for d in dir_inputs_x for file in d]
    target_y = [str(file) for d in dir_inputs_y for file in d]

    pool.map(worker, target_x + target_y)
    pool.close()
    pool.join()

    exit()  # TODO: logic

    m_dir_files_x, m_dir_files_y = [], []

    for d_x, d_y in zip(m_dir_files_x, m_dir_files_y):
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
