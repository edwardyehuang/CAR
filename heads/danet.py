import tensorflow as tf

from iseg.layers.model_builder import ConvBnRelu
from iseg.utils.attention_utils import *

from iseg.layers.self_attention import SelfAttention

from iseg.utils.common import resize_image

from iseg.vis.vismanager import get_visualization_manager
from heads.baseline import Baseline
from car_core.car import ClassAwareRegularization


class PositionAttentionModule(SelfAttention):
    def __init__(self, guided_filters=64, filters=512, name=None):
        super().__init__(guided_filters=guided_filters, filters=filters, name=name)

        self.gamma = self.add_weight("gamma", shape=(), initializer=tf.zeros_initializer())

    def call(self, inputs, training=None):

        vis_manager = get_visualization_manager()

        x = super().call(inputs, training=training)

        if vis_manager.recording:
            vis_manager.easy_add(x, "pam_im")

        x = self.gamma * x + inputs

        return x


class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(ChannelAttentionModule, self).__init__(name=name if name is not None else "ChannelAttentionModule")

        self.gamma = self.add_weight("gamma", shape=(), initializer=tf.zeros_initializer())

    def call(self, inputs, training=None):

        vis_manager = get_visualization_manager()

        inputs_shape = tf.shape(inputs)

        query = flatten_hw(inputs)
        key = transpose_hw_c(query)

        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

        attention = tf.matmul(key, query)  # Channel attention inverse [N, C, C]

        max_attention = tf.reduce_max(attention, axis=-1, keepdims=True)
        max_attention = tf.broadcast_to(max_attention, tf.shape(attention))

        attention = max_attention - attention

        attention = tf.nn.softmax(attention)

        value = tf.matmul(query, attention)
        value = tf.cast(value, inputs.dtype)
        value = tf.reshape(value, inputs_shape)

        if vis_manager.recording:
            vis_manager.easy_add(value, "cam_im")

        value = self.gamma * value + inputs

        return value


class DualAttentionHead(Baseline):
    def __init__(
        self, 
        use_aux_loss=True, 
        train_mode=False,
        baseline_mode=True,
        replace_2nd_last_conv=False,name=None):

        super().__init__(
            train_mode=train_mode, 
            baseline_mode=baseline_mode, 
            replace_2nd_last_conv=replace_2nd_last_conv, name=name,
            car_filters=512,
        )

        self.multi_loss = use_aux_loss

        inner_channels = 512

        self.pam_reduce_conv = ConvBnRelu(inner_channels, (3, 3), name="pam_reduce_conv")
        self.pam = PositionAttentionModule(guided_filters=64, filters=inner_channels)
        
        self.cam_reduce_conv = ConvBnRelu(inner_channels, (3, 3), name="cam_reduce_conv")
        self.cam = ChannelAttentionModule()
        
        if not self.replace_2nd_last_conv:
            self.pam_conv = ConvBnRelu(inner_channels, (3, 3), name="pam_conv")
            self.cam_conv = ConvBnRelu(inner_channels, (3, 3), name="cam_conv")

        if not baseline_mode:
            self.gta_c = ClassAwareRegularization(pooling_rates=[1], filters=512, name="gc")

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

        vis_manager = get_visualization_manager()

        pam = self.pam_reduce_conv(x, training=training)
        pam = self.pam(pam, training=training)
        cam = self.cam_reduce_conv(x, training=training)
        cam = self.cam(cam, training=training)
        
        if not self.baseline_mode:
            pam = self.gta((pam, label), training=training)
            cam = self.gta_c((cam, label), training=training)

        if not self.replace_2nd_last_conv:
            pam = self.pam_conv(pam, training=training)
            cam = self.cam_conv(cam, training=training)

        out = tf.add(pam, cam)

        if vis_manager.recording:
            vis_manager.easy_add(out, "danet_out")

        if self.multi_loss:
            return out, pam, cam
        else:
            return out
