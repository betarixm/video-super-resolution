import tensorflow as tf
import numpy as np

from core.utils import depth_to_space_3D


class FR_16(tf.keras.layers.Layer):
    def __init__(self, uf=4, **kwargs):
        super().__init__(**kwargs)
        self.uf = uf
        self.T_in = 7

    def build(self, batch_input_shape):
        F = 64
        G = 32

        self.conv1 = tf.keras.layers.Conv3D(64, kernel_size=(1, 3, 3), strides=1, padding="VALID")
        self.conv2 = tf.keras.layers.Conv3D(256, kernel_size=(1, 3, 3), strides=1, padding="VALID")
        self.RconvA = [tf.keras.layers.Conv3D(F + i * G, kernel_size=(1, 1, 1), strides=1, padding="VALID") for i in range(6)]
        self.RconvB = [tf.keras.layers.Conv3D(G, kernel_size=(3, 3, 3), strides=1, padding="VALID") for i in range(6)]
        self.rconv1 = tf.keras.layers.Conv3D(256, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.rconv2 = tf.keras.layers.Conv3D(3 * self.uf * self.uf, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.fconv1 = tf.keras.layers.Conv3D(512, kernel_size=(1, 1, 1), strides=1, padding="VALID")
        self.fconv2 = tf.keras.layers.Conv3D(1 * 5 * 5 * self.uf * self.uf, kernel_size=(1, 1, 1), strides=1, padding="VALID")

        self.RbnA = [tf.keras.layers.BatchNormalization() for i in range(6)]
        self.RbnB = [tf.keras.layers.BatchNormalization() for i in range(6)]
        self.fbn = tf.keras.layers.BatchNormalization()

        super().build(batch_input_shape)


    def call(self, inputs):

        stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
        sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

        x = self.conv1(tf.pad(inputs, sp, mode='CONSTANT'))
        for i in range(0, 3):
            t = self.RbnA[i](x)
            t = tf.keras.layers.Activation("relu")(t)
            t = self.RconvA[i](t)

            t = self.RbnB[i](t)
            t = tf.keras.layers.Activation("relu")(tf.pad(t, stp, mode='CONSTANT'))
            t = self.RconvB[i](t)

            x = tf.concat([x, t], 4)

        for i in range(3, 6):
            t = self.RbnA[i](x)
            t = tf.keras.layers.Activation("relu")(t)
            t = self.RconvA[i](t)

            t = self.RbnB[i](t)
            t = tf.keras.layers.Activation("relu")(t)
            t = self.RconvB[i](tf.pad(t, sp, mode='CONSTANT'))

            x = tf.concat([x[:, 1:-1], t], 4)

        x = self.fbn(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.conv2(tf.pad(x, sp, mode='CONSTANT'))
        x = tf.nn.relu(x)

        r = self.rconv1(x)
        r = tf.keras.layers.Activation("relu")(r)
        r = self.rconv2(r)

        f = self.fconv1(r)
        f = tf.keras.layers.Activation("relu")(f)
        f = self.fconv2(f)

        ds_f = tf.shape(f)
        f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, self.uf * self.uf])
        f = tf.nn.softmax(f, axis=4)

        return f, r

    class DynFilter(tf.keras.layers.Layer):

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


class DynFilter(tf.keras.layers.Layer):

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


class OurModel(tf.keras.Model):

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
