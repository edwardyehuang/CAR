import tensorflow as tf
import numpy as np

from iseg.layers.model_builder import ConvBnRelu
from iseg.utils.common import resize_image
from heads.baseline import Baseline

# This code is manually translated from offical repo
# https://github.com/speedinghzl/CCNet


class CrissCrossAttentionHead(Baseline):
    def __init__(
        self,
        num_recurrence=2,
        use_aux_loss=False,
        train_mode=False,
        baseline_mode=True,
        replace_2nd_last_conv=False,
        name=None,
    ):

        super().__init__(
            train_mode=train_mode, baseline_mode=baseline_mode, replace_2nd_last_conv=replace_2nd_last_conv, name=name
        )

        self.num_recurrence = num_recurrence
        self.use_aux_loss = use_aux_loss

        self.conv_a = ConvBnRelu(512, (3, 3), name="conv_a")
        self.conv_b = ConvBnRelu(512, (3, 3), name="conv_b")

        self.cc_head = CrissCrossAttentionBlock(name="CrissCrossAttentionBlock")

        if not self.replace_2nd_last_conv:
            self.bottleneck = ConvBnRelu(512, (3, 3), name="bottleneck")

        if self.use_aux_loss:
            self.dsn = ConvBnRelu(512, (3, 3), name="dsn")

    def call(self, inputs, training=None):

        label = None

        if self.train_mode:
            endpoints, label = inputs
        else:
            endpoints = inputs

        x = endpoints[-1]

        if self.train_mode:
            label = tf.cast(label, x.dtype)
            label = tf.expand_dims(label, axis=-1)  # [N, H, W, 1]
            label = resize_image(label, size=tf.shape(x)[1:3], method="nearest")  # [N, H, W, 1]

        x = self.conv_a(x, training=training)

        for _ in range(self.num_recurrence):
            x = self.cc_head(x, training=training)

        x = self.conv_b(x, training=training)
        x = tf.concat([endpoints[-1], x], axis=-1)

        if not self.baseline_mode:
            x = self.gta((x, label), training=training)
        if not self.replace_2nd_last_conv:
            x = self.bottleneck(x, training=training)

        if self.use_aux_loss:
            x = [x, self.dsn(endpoints[-2], training=training)]

        return x


class CrissCrossAttentionBlock(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)

    def build(self, input_shape):

        channels = input_shape[-1]

        self.query_conv = tf.keras.layers.Conv2D(channels // 8, (1, 1), name=f"{self.name}/query_conv")
        self.key_conv = tf.keras.layers.Conv2D(channels // 8, (1, 1), name=f"{self.name}/key_conv")
        self.value_conv = tf.keras.layers.Conv2D(channels, (1, 1), name=f"{self.name}/value_conv")

        self.gamma = self.add_weight("gamma", shape=(), initializer=tf.zeros_initializer())

        self.cc_attention = CrissCrossAttention(name=f"{self.name}/cc_attention")

    def call(self, inputs, training=None):

        x = inputs

        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        y = self.cc_attention([q, k, v], training=training) * self.gamma
        y += inputs

        return y


class CrissCrossAttention(tf.keras.Model):
    def __init__(self, name=None):

        super().__init__(name=name)

    def reduce_self(self, dot_product):

        field_size = tf.shape(dot_product)[-1]

        diag = tf.ones([field_size], dtype=dot_product.dtype) * tf.constant(np.inf, dtype=dot_product.dtype)  # [H]
        diag = tf.linalg.diag(diag)  # [H, H] rank = 2
        diag = tf.reshape(diag, [1, 1, field_size, field_size])  # [1, 1, H, W]

        return dot_product - diag

    def call(self, inputs, training=None):

        x = inputs  # [N, H, W, C]

        q = x[0]
        k = x[1]
        v = x[2]

        inputs_shape = tf.shape(v)
        height = inputs_shape[1]
        width = inputs_shape[2]

        vertial_q = tf.transpose(q, [0, 2, 1, 3])  # [N, W, H, C]
        vertial_k = tf.transpose(k, [0, 2, 3, 1])  # [N, W, C, H]

        vertial_a = tf.matmul(vertial_q, vertial_k)  # [N, W, H, H]
        vertial_a = self.reduce_self(vertial_a)
        vertial_a = tf.transpose(vertial_a, [0, 2, 1, 3])  # [N, H, W, H]

        horizontal_q = q  # [N, H, W, C]
        horizontal_k = tf.transpose(k, [0, 1, 3, 2])  # [N, H, C, W]

        horizontal_a = tf.matmul(horizontal_q, horizontal_k)  # [N, H, W, W]

        a = tf.concat([vertial_a, horizontal_a], axis=-1)  # [N, H, W, H + W]
        a = tf.nn.softmax(a)  # [N, H, W, H + W]

        vertial_a = a[:, :, :, 0:height]  # [N, H, W, H]
        vertial_a = tf.transpose(vertial_a, [0, 2, 1, 3])  # [N, W, H, H]
        v = tf.transpose(v, [0, 2, 1, 3])  # [N, W, H, C]
        vertial_y = tf.matmul(vertial_a, v)  # [N, W, H, C]
        vertial_y = tf.transpose(vertial_y, [0, 2, 1, 3])  # [N, H, W, C]

        horizontal_a = a[:, :, :, height : height + width]  # [N, H, W, W]
        v = tf.transpose(v, [0, 2, 1, 3])  # [N, H, W, C]
        horizontal_y = tf.matmul(horizontal_a, v)  # [N, H, W, C]

        y = vertial_y + horizontal_y

        return y
