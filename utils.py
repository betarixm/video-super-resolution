from PIL import Image
import numpy as np


def load_image(path, channel_mean=None, mod_crop=None):
    if mod_crop is None:
        mod_crop = [0, 0, 0, 0]

    img = Image.open(path)
    cimg = img.convert('RGB')
    x = np.asarray(cimg, dtype='float32')
    x *= 1.0/255.0

    if channel_mean:
        x[:, :, 0] -= channel_mean[0]
        x[:, :, 1] -= channel_mean[1]
        x[:, :, 2] -= channel_mean[2]

    if mod_crop[0] * mod_crop[1] * mod_crop[2] * mod_crop[3]:
        x = x[mod_crop[0]:-mod_crop[1], mod_crop[2]:-mod_crop[3], :]

    return x
