import tensorflow as tf

from iseg.layers.model_builder import ConvBnRelu
from iseg.utils.attention_utils import *

from iseg.utils.common import resize_image

from heads.baseline import Baseline

# Note that, we reviewed many implementations of FCN, including the one in mmsegmentation
# We choose current one to keep it simple and fair.


class FCNHead(Baseline):
    def __init__(
        self, use_aux_loss=False, train_mode=False, baseline_mode=True, replace_2nd_last_conv=False, name=None,
    ):

        super().__init__(
            train_mode=train_mode, baseline_mode=baseline_mode, replace_2nd_last_conv=replace_2nd_last_conv, name=name
        )

        self.use_aux_loss = use_aux_loss
        self.train_mode = train_mode

        self.down_conv = ConvBnRelu(512, (3, 3), name="down_conv")

        if not self.replace_2nd_last_conv:
            self.end_conv = ConvBnRelu(256, (1, 1), name="end_conv")

    def call(self, inputs, training=None):

        label = None

        if self.train_mode:
            endpoints, label = inputs
        else:
            endpoints = inputs

        x = endpoints[-1]

        x = self.down_conv(x, training=training)

        if self.train_mode:
            label = tf.cast(label, x.dtype)
            label = tf.expand_dims(label, axis=-1)  # [N, H, W, 1]
            label = resize_image(label, size=tf.shape(x)[1:3], method="nearest")  # [N, H, W, 1]

        if not self.baseline_mode:
            x = self.gta((x, label), training=training)

        if not self.replace_2nd_last_conv:
            x = self.end_conv(x, training=training)

        return x
