## -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np

from utils import *
from nets import *


class OurModel(keras.Model):

    def __init__(self, uf=4, **kwargs):

        super().__init__(**kwargs)

        F = 64
        G = 32

        self.uf = 4
        self.conv1 = keras.layers.Conv3D(64, kernel_size=(1, 3, 3), strides=1, padding="VALID")
        self.conv2 = keras.layers.Conv3D(256, kernel_size=(1, 3, 3), strides=1, padding="VALID")
        self.RconvA = [keras.layers.Conv3D(F + i * G, kernel_size=(1, 1, 1), strides=1, padding="VALID") for i in range(6)]
        self.RconvB = [keras.layers.Conv3D(G, kernel_size=(3, 3, 3), strides=1, padding="VALID") for i in range(6)]
        self.rconv1 = keras.layers.Conv3D(256, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.rconv2 = keras.layers.Conv3D(3 * uf * uf, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.fconv1 = keras.layers.Conv3D(512, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.fconv2 = keras.layers.Conv3D(1 * 5 * 5 * uf * uf, kernel_size=(1, 1, 1), strides=1, padding="VALID")

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
            t = keras.layers.BatchNormalization()(x)
            t = keras.layers.Activation("relu")(t)
            t = self.RconvA[i](t)

            t = keras.layers.BatchNormalization()(t)
            t = keras.layers.Activation("relu")(tf.pad(t, stp, mode='CONSTANT'))
            t = self.RconvB[i](t)

            x = tf.concat([x, t], 4)

        for i in range(3, 6):
            t = keras.layers.BatchNormalization()(x)
            t = keras.layers.Activation("relu")(t)
            t = self.RconvA[i](t)

            t = keras.layers.BatchNormalization()(t)
            t = keras.layers.Activation("relu")(t)
            t = self.RconvB[i](tf.pad(t, sp, mode='CONSTANT'))

            x = tf.concat([x[:, 1:-1], t], 4)

        x = keras.layers.BatchNormalization()(x)
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

        return self.G(inputs, f, r, 4, 7)


def train_and_evaluate():
    pass


if __name__ == "__main__":
    model = OurModel()
    model.build(input_shape=[1,7,32,32,3])
    model.summary()



