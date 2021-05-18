import torch
from PIL import Image
import numpy as np
import tensorflow as tf


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


def down_sample(x, h, scale=4):
    # TODO: Modify Logic to Use Pytorch
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], 3])

    # Reflect padding
    w = tf.constant(h)

    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]

    depthwise_f = tf.tile(w, [1, 1, 3, 1])
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_f, [1, scale, scale, 1], 'VALID')

    ds_y = tf.shape(y)
    y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return torch.from_numpy(y.numpy())


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


def to_uint8(x: np.array, _min, _max):
    x = x.astype('float32')
    x = (x - _min) / (_max - _min) * 255
    return np.clip(np.round(x), 0, 255)


def avg_psnr(vid_true, vid_pred, _min=0, _max=255, t_border=2, sp_border=8, is_t_y=False, is_p_y=False):
    input_shape = vid_pred.shape
    if is_t_y:
        y_true = to_uint8(vid_true, _min, _max)
    else:
        y_true = np.empty(input_shape[:-1])
        for t in range(input_shape[0]):
            y_true[t] = rgb_to_ycbcr(to_uint8(vid_true[t], _min, _max), 255)[:, :, 0]

    if is_p_y:
        y_pred = to_uint8(vid_pred, _min, _max)
    else:
        y_pred = np.empty(input_shape[:-1])
        for t in range(input_shape[0]):
            y_pred[t] = rgb_to_ycbcr(to_uint8(vid_pred[t], _min, _max), 255)[:, :, 0]

    diff = y_true - y_pred
    diff = diff[t_border: input_shape[0]- t_border, sp_border: input_shape[1]- sp_border, sp_border: input_shape[2]- sp_border]

    psnrs = []
    for t in range(diff.shape[0]):
        rmse = np.sqrt(np.mean(np.power(diff[t],2)))
        psnrs.append(20*np.log10(255./rmse))

    return np.mean(np.asarray(psnrs))


def batch_norm(i, is_train, decay=0.999, name='BatchNorm'):
    raise Exception("Use torch.BatchNorm")

# TODO: Conv3D compat

# TODO: Use torch.load(PATH) instead of LoadParams


def depth_to_space_3d(x: torch.Tensor, block_size):
    ds_x = x.shape
    x = torch.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], ds_x[4]])
    y = torch.pixel_shuffle(x, block_size)  # Check that input is NCHW format
    ds_y = y.shape
    return torch.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])

