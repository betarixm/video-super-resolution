#!/usr/bin/env python

# https://github.com/yhjo09/VSR-DUF/blob/master/utils.py
import tensorflow as tf
from PIL import Image
import numpy as np
import h5py


def LoadImage(path, color_mode='RGB', channel_mean=None, mod_crop=None):
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


def DownSample(x, h, scale=4):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], 3])

    # Reflect padding
    W = tf.constant(h)

    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    t = tf.constant([1, 1, 3, 1])
    depthwise_f = tf.tile(W, t)
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_f, [1, scale, scale, 1], 'VALID')

    ds_y = tf.shape(y)
    y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y


def _rgb2ycbcr(img, max_val=255):
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
            y_true[t] = _rgb2ycbcr(to_uint8(vid_true[t], _min, _max), 255)[:, :, 0]

    if is_p_y:
        y_pred = to_uint8(vid_pred, _min, _max)
    else:
        y_pred = np.empty(input_shape[:-1])
        for t in range(input_shape[0]):
            y_pred[t] = _rgb2ycbcr(to_uint8(vid_pred[t], _min, _max), 255)[:, :, 0]

    diff = y_true - y_pred
    diff = diff[t_border: input_shape[0]- t_border, sp_border: input_shape[1]- sp_border, sp_border: input_shape[2]- sp_border]

    psnrs = []
    for t in range(diff.shape[0]):
        rmse = np.sqrt(np.mean(np.power(diff[t],2)))
        psnrs.append(20*np.log10(255./rmse))

    return np.mean(np.asarray(psnrs))


def BatchNorm(x, is_train, decay=0.999, name='BatchNorm'):
    '''
    https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops

    axis = list(range(len(x.get_shape()) - 1))
    fdim = x.get_shape()[-1:]

    # TODO: Use tf.Variable instead of compat.v1
    # net.py
    # beta = tf.Variable(tf.constant_initializer(value=0.0), shape=fdim, name="beta")
    # gamma = tf.Variable(tf.constant_initializer(value=1.0), shape=fdim, name="gamma")
    # moving_mean = tf.Variable(tf.constant_initializer(value=0.0), shape=fdim, trainable=False, name="moving_mean")
    # moving_variance = tf.Variable(tf.constant_initializer(value=0.0), shape=fdim, trainable=False, name="moving_variance")
    #
    # utils.py
    # def BatchNorm(beta, gamma, moving_mean, moving_variance, ...)
    # ...
    # Delete below codes
    with tf.compat.v1.variable_scope(name):
        beta = tf.compat.v1.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.compat.v1.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.compat.v1.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        moving_variance = tf.compat.v1.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)

        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-3) #, tf.stack([mean[0], variance[0], beta[0], gamma[0]])


def Conv3D(x, kernel_shape, strides, padding, name='Conv3d', w_initializer=tf.keras.initializers.HeNormal(), bias=True):
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable("W", kernel_shape, initializer=w_initializer)
        if bias is True:
            b = tf.compat.v1.get_variable("b", (kernel_shape[-1]),initializer=tf.constant_initializer(value=0.0))
        else:
            b = 0

    return tf.nn.conv3d(x, W, strides, padding) + b


def LoadParams(sess, params, in_file='params.hdf5'):
    raise Exception("Use keras.models.load_model")


def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.nn.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


def DynFilter3D(x, f, filter_size):
    """
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    """
    # make tower
    filter_local_expand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_local_expand = tf.Variable(filter_local_expand_np, trainable=False, dtype='float32', name='filter_localexpand')
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    x_local_expand = tf.nn.conv2d(x, filter_local_expand, [1, 1, 1, 1], 'SAME')  # b, h, w, 1*5*5
    x_local_expand = tf.expand_dims(x_local_expand, axis=3)  # b, h, w, 1, 1*5*5
    x = tf.matmul(x_local_expand, f)  # b, h, w, 1, R*R
    x = tf.squeeze(x, axis=3)  # b, h, w, R*R

    return x


def Huber(y_true, y_pred, delta, axis=None):
    abs_error = tf.abs(y_pred - y_true)
    quadratic = tf.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses, axis=axis)

