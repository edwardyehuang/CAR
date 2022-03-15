import tensorflow as tf

from iseg.layers.model_builder import ConvBnRelu, ImageLevelBlock
from iseg.utils.attention_utils import *

from iseg.utils.common import resize_image

from heads.baseline import Baseline

# This code is manually translated from offical repo
# https://github.com/tensorflow/models/tree/master/research/deeplab


class DeepLabv3Head(Baseline):
    def __init__(
        self,
        use_aux_loss=False,
        train_mode=False,
        baseline_mode=True,
        replace_2nd_last_conv=False,
        output_stride=8,
        name=None,
    ):

        super().__init__(
            train_mode=train_mode,
            baseline_mode=baseline_mode,
            replace_2nd_last_conv=replace_2nd_last_conv,
            car_filters=256,
            name=name,
        )

        self.train_mode = train_mode
        self.output_stride = output_stride

        rate = int(32 / output_stride)

        self.aspp = AtrousSpatialPyramidPooling(
            filters=256, receptive_fields=[3 * rate, 6 * rate, 9 * rate], pixel_level=True, image_level=True,
        )  # Default setups in DeepLab V3

        if not self.replace_2nd_last_conv:
            self.end_conv = ConvBnRelu(256, (1, 1), name="end_conv")

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

        x = self.aspp(x, training=training)

        if not self.baseline_mode:
            x = self.gta((x, label), training=training)

        if not self.replace_2nd_last_conv:
            x = self.end_conv(x, training=training)

        return x


class AtrousSpatialPyramidPooling(tf.keras.Model):
    def __init__(self, filters=256, receptive_fields=[3, 6, 9], pixel_level=True, image_level=True, name=None):

        super(AtrousSpatialPyramidPooling, self).__init__(
            name=name if name is not None else "AtrousSpatialPyramidPooling"
        )

        self.pixel_level = pixel_level
        self.image_level = image_level

        if self.image_level:
            self.image_level_block = ImageLevelBlock(filters, name="image_level_block")

        if self.pixel_level:
            self.pixel_level_block = ConvBnRelu(filters, (1, 1), name="pixel_level_block")

        self.middle_convs = []
        self.middle_convs_bn = []

        for rate in receptive_fields:
            self.middle_convs.append(ConvBnRelu(filters, (3, 3), dilation_rate=rate, name="rate_{}_conv".format(rate)))

    def call(self, inputs, training=None):

        results = []

        if self.image_level:
            results.append(self.image_level_block(inputs, training=training))

        if self.pixel_level:
            results.append(self.pixel_level_block(inputs, training=training))

        for i in range(len(self.middle_convs)):
            results.append(self.middle_convs[i](inputs, training=training))

        results = tf.concat(results, axis=-1)

        return results
