import multiprocessing
from PIL import Image
import numpy as np

NUM_DIR = 339

manager = multiprocessing.Manager()
m_dir_lr_files = [manager.dict() for _ in range(NUM_DIR)]
m_dir_hr_files = [manager.dict() for _ in range(NUM_DIR)]


def worker(path: str):
    print("[S]" + path, flush=True)

    color_mode, channel_mean, mod_crop='RGB', None, [0, 0, 0, 0]

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

    if channel_mean is not None:
        x[:, :, 0] -= channel_mean[0]
        x[:, :, 1] -= channel_mean[1]
        x[:, :, 2] -= channel_mean[2]

    if mod_crop[0] * mod_crop[1] * mod_crop[2] * mod_crop[3]:
        x = x[mod_crop[0]:-mod_crop[1], mod_crop[2]:-mod_crop[3], :]

    p = path.split("/")
    category, num_dir, num_file = p[-3:]

    if category == "HR":
        m_dir_hr_files[int(num_dir)][int(num_file.replace(".png", ""))] = x
    elif category == "LR":
        m_dir_lr_files[int(num_dir)][int(num_file.replace(".png", ""))] = x

    print("[D]" + path, flush=True)
