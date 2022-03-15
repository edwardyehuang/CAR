import tensorflow as tf
import numpy as np

from iseg.layers.normalizations import normalization
from iseg.layers.model_builder import ConvBnRelu, get_tensor_shape
from iseg.utils.attention_utils import flatten_hw
from iseg.vis.vismanager import get_visualization_manager
from heads.baseline import Baseline


class AggregationModule(tf.keras.Model):
    def __init__(self, filters=512, kernel_size=11, name=None):
        super().__init__(name=name)

        self.reduce_conv = ConvBnRelu(filters=filters, kernel_size=(3, 3), name="reduce_conv")

        self.t1 = tf.keras.layers.Conv2D(filters, (kernel_size, 1), groups=filters, padding="same", name="t1")
        self.t2 = tf.keras.layers.Conv2D(filters, (1, kernel_size), groups=filters, padding="same", name="t2")

        self.p1 = tf.keras.layers.Conv2D(filters, (1, kernel_size), groups=filters, padding="same", name="p1")
        self.p2 = tf.keras.layers.Conv2D(filters, (kernel_size, 1), groups=filters, padding="same", name="p2")

        self.norm = normalization(name="norm")

    def call(self, inputs, training=None):

        x = self.reduce_conv(inputs, training=training)

        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        out = tf.nn.relu(self.norm(x1 + x2, training=training))

        return out


class ContextPriorHead(Baseline):
    def __init__(
        self, 
        prior_size=64, 
        prior_channels=512, 
        use_aux_loss=False, 
        num_class=21,
        train_mode=False,
        baseline_mode=True,
        replace_2nd_last_conv=False,
        name=None):

        super().__init__(
            train_mode=train_mode, 
            baseline_mode=baseline_mode, 
            replace_2nd_last_conv=replace_2nd_last_conv,
            name=name)

        self.prior_size = [prior_size, prior_size]
        self.num_class = num_class

        self.aggregation = AggregationModule(filters=prior_channels)
        self.prior_conv = ConvBnRelu(
            filters=np.prod(self.prior_size),
            kernel_size=1, 
            activation=None, 
            name="prior_conv"
            )

        self.intra_conv = ConvBnRelu(filters=prior_channels, name="intra_conv")
        self.inter_conv = ConvBnRelu(filters=prior_channels, name="inter_conv")

        if not self.replace_2nd_last_conv:
            self.bottleneck = ConvBnRelu(512, (3, 3), name="bottleneck")

        self.use_aux_loss = use_aux_loss
        self.logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name=f"logits_conv")

        self.vis_manager = get_visualization_manager()


    def call(self, inputs, training=None):

        label = None

        if self.train_mode:
            endpoints, label = inputs
        else:
            endpoints = inputs

        x = endpoints[-1]

        x = tf.image.resize(x, size=self.prior_size, method="bilinear") # Make sure image size == prior size
        x = tf.cast(x, endpoints[-1].dtype)

        if self.train_mode:
            label = tf.cast(label, x.dtype)
            label = tf.expand_dims(label, axis=-1)  # [N, H, W, 1]
            label = tf.image.resize(label, size=tf.shape(x)[1:3], method="nearest")  # [N, H, W, 1]

        backbone_idenitiy = tf.identity(x, name="backbone_identity")

        x = self.aggregation(x, training=training)  # [N, H, W, 512]

        batch_size, height, width, channels = get_tensor_shape(x)

        value = tf.identity(x, name="value")
        value = tf.reshape(value, [batch_size, self.prior_size[0] * self.prior_size[1], channels])  # [N, 4096, 512]

        context_prior_map = self.prior_conv(x, training=training)  # [N, H, W, 4096]
        context_prior_map = flatten_hw(context_prior_map)

        context_prior_map = tf.sigmoid(context_prior_map) # [N, HW, 4096]

        inter_context_prior_map = 1 - context_prior_map # [N, HW, 4096]

        intra_context = tf.matmul(context_prior_map, value)  # [N, HW, 512]
        intra_context /= np.prod(self.prior_size)
        intra_context = tf.reshape(
            intra_context, [batch_size, height, width, channels]
        )  # [N, 64, 64, 512]
        intra_context = self.intra_conv(intra_context, training=training)

        inter_context = tf.matmul(inter_context_prior_map, value)  # [N, 4096, 512]
        inter_context /= np.prod(self.prior_size)
        inter_context = tf.reshape(
            inter_context, [batch_size, height, width, channels]
        )  # [N, 64, 64 512]
        inter_context = self.inter_conv(inter_context, training=training)

        context_prior_outs = tf.concat([backbone_idenitiy, intra_context, inter_context], axis=-1)  # [N, H, HW, 3 * 512]

        if not self.baseline_mode:
            output = self.gta((context_prior_outs, label), training=training)
        if not self.replace_2nd_last_conv:
            output = self.bottleneck(context_prior_outs, training=training)

        if self.vis_manager.recording:
            self.vis_manager.easy_add(output, name="gta_rich")

        output = self.logits_conv(output)
        output = tf.cast(output, tf.float32)

        if self.use_aux_loss:
            context_prior_map = tf.reshape(context_prior_map, (batch_size, height, width, context_prior_map.shape[-1]))
            context_prior_map = tf.cast(context_prior_map, tf.float32)
            output = [output, context_prior_map]

        return output


def construct_ideal_affinity_martix (label, label_size, ignore_label=255, num_class=21):

    label = tf.expand_dims(label, axis=-1) # [N, H, W, 1]
    label = tf.image.resize(label, size=(label_size[0], label_size[1]), method="nearest")
    label = tf.squeeze(label, axis=-1) # [N, H, W]
    label = tf.cast(label, tf.int32) # [N, H, W]

    if ignore_label > num_class:
        label = tf.where(label == ignore_label, num_class, label)

    one_hot_labels = tf.one_hot(label, num_class + 1) # [N, H, W, class + 1]

    one_hot_label_shape = tf.shape(one_hot_labels)
    
    one_hot_labels = tf.reshape(one_hot_labels, (
        one_hot_label_shape[0],
        one_hot_label_shape[1] * one_hot_label_shape[2],
        num_class + 1)) # [N, HW, class + 1]

    ideal_affinity_martix = tf.matmul(one_hot_labels, tf.transpose(one_hot_labels, [0, 2, 1])) # [N, HW, HW]

    return ideal_affinity_martix


def affinity_loss (
    num_class=21,
    ignore_label=255,
    batch_size=2,
    reduction=False,
    from_logits=True,
    **kwargs,
):

    loss_func = tf.keras.losses.BinaryCrossentropy(
        from_logits=from_logits,
        reduction=tf.keras.losses.Reduction.NONE
    )

    def weighted_loss(y_true, y_pred):
        # y_true : label [N, H, W]
        # y_pred : context_prior_map [N, H, W, HW]
    
        prior_size = tf.sqrt(tf.cast(y_pred.shape[-1], tf.float32))

        context_prior_map_shape = tf.shape(y_pred)
        ideal_affinity_martix = construct_ideal_affinity_martix(
            y_true, 
            context_prior_map_shape[1:3], 
            ignore_label=ignore_label, 
            num_class=num_class
            ) # [N, HW, HW]

        cls_score = tf.reshape(y_pred, (
            context_prior_map_shape[0],
            context_prior_map_shape[1] * context_prior_map_shape[2],
            y_pred.shape[-1])) # [N, HW, HW]

        unary_term = loss_func(cls_score, ideal_affinity_martix)

        diagonal_matrix = (1 - tf.eye(tf.shape(ideal_affinity_martix)[-1])) # [HW, HW]
        diagonal_matrix = tf.expand_dims(diagonal_matrix, axis=0) # [1, HW, HW]
        v_targets = diagonal_matrix * ideal_affinity_martix # [N, HW, HW]

        recall_part = tf.reduce_sum(cls_score * v_targets, axis=-1) # [N, HW]
        denominator = tf.reduce_sum(ideal_affinity_martix, axis=-1) # [N, HW]
        denominator = tf.where(denominator <= 0, 1.0, denominator) # [N, HW]
        recall_part /= denominator # [N, HW]
        recall_loss = loss_func(tf.ones_like(recall_part), recall_part)

        spec_part = (1 - cls_score) * (1 - ideal_affinity_martix) # [N, HW, HW]
        spec_part = tf.reduce_sum(spec_part, axis=-1) # [N, HW]
        denominator = tf.reduce_sum(1 - ideal_affinity_martix, axis=-1) # [N, HW]
        denominator = tf.where(denominator <= 0, 1.0, denominator) # [N, HW]
        spec_part /= denominator
        spec_loss = loss_func(tf.ones_like(spec_part), spec_part)

        precision_part = tf.reduce_sum(cls_score * ideal_affinity_martix, axis=-1) # [N, HW]
        denominator = tf.reduce_sum(cls_score, axis=-1)
        denominator = tf.where(denominator <= 0, 1.0, denominator) # [N, HW]
        precision_part /= denominator
        precision_loss = loss_func(tf.ones_like(precision_part), precision_part)

        global_term = recall_loss + spec_loss + precision_loss
        global_term /= tf.cast(prior_size, tf.float32)

        loss_cls = unary_term + tf.expand_dims(global_term, axis=-1)

        if reduction:
            loss_cls = tf.nn.compute_average_loss(loss_cls, global_batch_size=batch_size)

        return loss_cls

    return weighted_loss

