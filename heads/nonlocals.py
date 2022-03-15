import tensorflow as tf
import iseg.static_strings as ss

from iseg.layers.model_builder import ConvBnRelu
from iseg.utils.attention_utils import *
from iseg.vis.vismanager import get_visualization_manager
from iseg.utils.common import resize_image

from iseg.layers.self_attention import SelfAttention

from heads.baseline import Baseline


class LabelNonLocalHead(Baseline):
    def __init__(
        self, use_aux_loss=False, train_mode=False, baseline_mode=True, replace_2nd_last_conv=False, name=None,
    ):

        super().__init__(
            train_mode=train_mode, baseline_mode=baseline_mode, replace_2nd_last_conv=replace_2nd_last_conv, name=name
        )

        self.use_aux_loss = use_aux_loss

        self.down_conv = ConvBnRelu(512, (3, 3), name="down_conv")
        self.attention = SelfAttention(guided_filters=64, filters=512, shared_querykey_weights=True)

        if not self.replace_2nd_last_conv:
            self.end_conv = ConvBnRelu(512, (3, 3), name="end_conv")  # Most commonly used for reguluar non-locals
        elif self.baseline_mode and self.replace_2nd_last_conv:
            self.end_conv = ConvBnRelu(512, (1, 1), name="end_conv")  # For ablation stuides only

        if self.use_aux_loss:
            self.aux_down_conv = ConvBnRelu(512, (3, 3), name="down_conv")

        self.vis_manager = get_visualization_manager()

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

        x = self.down_conv(x, training=training)

        a = self.attention(x, training=training)
        x += a

        if not self.baseline_mode:
            x = self.gta((x, label), training=training)

        if not self.replace_2nd_last_conv or (self.baseline_mode and self.replace_2nd_last_conv):
            x = self.end_conv(x, training=training)

        if self.vis_manager.recording:
            self.vis_manager.easy_add(x, name="gta_rich")

        if self.use_aux_loss:
            aux_x = endpoints[-2]
            aux_x = self.aux_down_conv(aux_x, training=training)

            x = [x, aux_x]

        return x
