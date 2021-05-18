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


def rgb_to_ycbcr(img, max_val=255):
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if max_val == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr
