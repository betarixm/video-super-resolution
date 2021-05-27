## -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob

from utils import *


class FR_16(keras.layers.Layer):

    def __init__(self, uf=4, **kwargs):

        super().__init__(**kwargs)
        self.uf = uf
        self.T_in = 7

    def build(self, batch_input_shape):

        F = 64
        G = 32

        self.conv1 = keras.layers.Conv3D(64, kernel_size=(1, 3, 3), strides=1, padding="VALID")
        self.conv2 = keras.layers.Conv3D(256, kernel_size=(1, 3, 3), strides=1, padding="VALID")
        self.RconvA = [keras.layers.Conv3D(F + i * G, kernel_size=(1, 1, 1), strides=1, padding="VALID") for i in range(6)]
        self.RconvB = [keras.layers.Conv3D(G, kernel_size=(3, 3, 3), strides=1, padding="VALID") for i in range(6)]
        self.rconv1 = keras.layers.Conv3D(256, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.rconv2 = keras.layers.Conv3D(3 * self.uf * self.uf, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.fconv1 = keras.layers.Conv3D(512, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.fconv2 = keras.layers.Conv3D(1 * 5 * 5 * self.uf * self.uf, kernel_size=(1, 1, 1), strides=1, padding="VALID")

        self.RbnA = [keras.layers.BatchNormalization() for i in range(6)]
        self.RbnB = [keras.layers.BatchNormalization() for i in range(6)]
        self.fbn = keras.layers.BatchNormalization()

        super().build(batch_input_shape)


    def G(self, x, Fx, Rx, R, T_in):
        # shape of x: [B,T_in,H,W,C]
        # Fx: [B,1,H,W,1*5*5,R*R]   filters
        # Rx: [B,1,H,W,3*R*R]   residual

        x_c = []
        for c in range(3):
            t = DynFilter3D(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
            t = tf.nn.depth_to_space(t, R)  # [B,H*R,W*R,1]
            x_c += [t]
        x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3]
        x = tf.expand_dims(x, axis=1)

        Rx = depth_to_space_3D(Rx, R)  # [B,1,H*R,W*R,3]
        x += Rx

        return x

    def call(self, inputs):

        stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
        sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

        x = self.conv1(tf.pad(inputs, sp, mode='CONSTANT'))
        for i in range(0, 3):
            t = self.RbnA[i](x)
            t = keras.layers.Activation("relu")(t)
            t = self.RconvA[i](t)

            t = self.RbnB[i](t)
            t = keras.layers.Activation("relu")(tf.pad(t, stp, mode='CONSTANT'))
            t = self.RconvB[i](t)

            x = tf.concat([x, t], 4)

        for i in range(3, 6):
            t = self.RbnA[i](x)
            t = keras.layers.Activation("relu")(t)
            t = self.RconvA[i](t)

            t = self.RbnB[i](t)
            t = keras.layers.Activation("relu")(t)
            t = self.RconvB[i](tf.pad(t, sp, mode='CONSTANT'))

            x = tf.concat([x[:, 1:-1], t], 4)

        x = self.fbn(x)
        x = keras.layers.Activation("relu")(x)
        x = self.conv2(tf.pad(x, sp, mode='CONSTANT'))
        x = tf.nn.relu(x)

        r = self.rconv1(x)
        r = keras.layers.Activation("relu")(r)
        r = self.rconv2(r)

        f = self.fconv1(r)
        f = keras.layers.Activation("relu")(f)
        f = self.fconv2(f)

        ds_f = tf.shape(f)
        f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, self.uf * self.uf])
        f = tf.nn.softmax(f, axis=4)

        return f, r


class DynFilter(keras.layers.Layer):

    def __init__(self, filter_size, **kwargs):
        """
        :param f: (b, h, w, tower_depth, output_depth)
        :param filter_size: filter_shape (ft, fh, fw)
        """

        super().__init__(**kwargs)

        self.filter_size = filter_size


    def build(self, batch_input_shape):

        # make tower
        self.filter_local_expand_np = np.reshape(np.eye(np.prod(self.filter_size), np.prod(self.filter_size)),
                                            (self.filter_size[1], self.filter_size[2], self.filter_size[0],
                                             np.prod(self.filter_size)))
        self.filter_local_expand = tf.Variable(self.filter_local_expand_np, trainable=False, dtype='float32',
                                          name='filter_localexpand')

        super().build(batch_input_shape)


    def call(self, x, **kwargs):
        """
        :param x: (b, t, h, w)
        :return:
        """

        f = kwargs['filter']
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x_local_expand = tf.nn.conv2d(x, self.filter_local_expand, [1, 1, 1, 1], 'SAME')  # b, h, w, 1*5*5
        x_local_expand = tf.expand_dims(x_local_expand, axis=3)  # b, h, w, 1, 1*5*5
        x = tf.matmul(x_local_expand, f)  # b, h, w, 1, R*R
        x = tf.squeeze(x, axis=3)  # b, h, w, R*R

        return x


class OurModel(keras.Model):

    def __init__(self, T_in=7, upscale_factor=4):

        super().__init__()

        self.T_in = T_in
        self.upscale_factor = upscale_factor

    def build(self, batch_input_shape):

        self.FR_16 = FR_16()
        self.DynFilter = [DynFilter([1, 5, 5]) for c in range(3)]

        super().build(batch_input_shape)

    def call(self, inputs):
        f, r = self.FR_16(inputs)

        #return self.DynFilter(inputs, filter=f)
        x_c = []
        for c in range(3):
            t = self.DynFilter[c](inputs[:, self.T_in // 2:self.T_in // 2 + 1, :, :, c], filter=f[:, 0, :, :, :, :])  # [B,H,W,R*R]
            t = tf.nn.depth_to_space(t, self.upscale_factor)  # [B,H*R,W*R,1]
            x_c += [t]
        x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3]
        x = tf.expand_dims(x, axis=1)

        r = depth_to_space_3D(r, self.upscale_factor)  # [B,1,H*R,W*R,3]
        x += r

        return x


def train_and_evaluate():
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
        for i in range(len(dir)-6):
            X_train.append(dir[i:i+7])
    for dir in dir_files_y:
        for i in range(len(dir)-6):
            y_train.append(dir[i+3])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    model = OurModel()
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.Huber(delta=0.01))
    history = model.fit(X_train, y_train, epochs = 1)
    model.save("./FR_16(4)", save_format="tf")


if __name__ == "__main__":
    train_and_evaluate()



