
# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
from iseg.layers.normalizations import normalization

from iseg.utils.attention_utils import *
from iseg.layers.model_builder import resize_image, get_training_value
from iseg.vis.vismanager import get_visualization_manager

from car_core.utils import (
    get_flatten_one_hot_label,
    get_class_sum_features_and_counts,
    get_inter_class_relative_loss,
    get_intra_class_absolute_loss,
    get_pixel_inter_class_relative_loss,
)


class ClassAwareRegularization(tf.keras.Model):
    def __init__(
        self,
        train_mode=False,
        use_inter_class_loss=True,
        use_intra_class_loss=True,
        intra_class_loss_remove_max=False,
        use_inter_c2c_loss=True,
        use_inter_c2p_loss=False,
        intra_class_loss_rate=1,
        inter_class_loss_rate=1,
        num_class=21,
        ignore_label=0,
        pooling_rates=[1],
        use_batch_class_center=True,
        use_last_class_center=False,
        last_class_center_decay=0.9,
        inter_c2c_loss_threshold=0.5,
        inter_c2p_loss_threshold=0.25,
        filters=512,
        apply_convs=False,
        name=None,
    ):

        super().__init__(name=name)

        self.vis_manager = get_visualization_manager()

        self.train_mode = train_mode
        self.use_inter_class_loss = use_inter_class_loss
        self.use_intra_class_loss = use_intra_class_loss
        self.intra_class_loss_rate = intra_class_loss_rate
        self.inter_class_loss_rate = inter_class_loss_rate
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.inter_c2c_loss_threshold = inter_c2c_loss_threshold
        self.inter_c2p_loss_threshold = inter_c2p_loss_threshold

        self.intra_class_loss_remove_max = intra_class_loss_remove_max


        self.use_inter_c2c_loss = use_inter_c2c_loss
        self.use_inter_c2p_loss = use_inter_c2p_loss

        self.filters = filters
        self.apply_convs = apply_convs

        if isinstance(pooling_rates, tuple):
            pooling_rates = list(pooling_rates)

        if not isinstance(pooling_rates, list):
            pooling_rates = [pooling_rates]

        self.pooling_rates = pooling_rates
        self.use_batch_class_center = use_batch_class_center
        self.use_last_class_center = use_last_class_center
        self.last_class_center_decay = last_class_center_decay

        print(f"------CAR settings------")
        print(f"------train_mode = {train_mode}")
        print(f"------use_intra_class_loss = {use_intra_class_loss}")
        print(f"------use_inter_class_loss = {use_inter_class_loss}")
        print(f"------intra_class_loss_rate = {intra_class_loss_rate}")
        print(f"------inter_class_loss_rate = {inter_class_loss_rate}")

        print(f"------use_batch_class_center = {use_batch_class_center}")
        print(f"------use_last_class_center = {use_last_class_center}")
        print(f"------last_class_center_decay = {last_class_center_decay}")

        print(f"------pooling_rates = {pooling_rates}")
        print(f"------inter_c2c_loss_threshold = {inter_c2c_loss_threshold}")
        print(f"------inter_c2p_loss_threshold = {inter_c2p_loss_threshold}")

        print(f"------intra_class_loss_remove_max = {intra_class_loss_remove_max}")

        print(f"------use_inter_c2c_loss = {use_inter_c2c_loss}")
        print(f"------use_inter_c2p_loss = {use_inter_c2p_loss}")

        print(f"------filters = {filters}")
        print(f"------apply_convs = {apply_convs}")

        print(f"------num_class = {num_class}")
        print(f"------ignore_label = {ignore_label}")



    def add_car_losses(self, features, label=None, extra_prefix=None, training=None):

        # features : [N, H, W, C]

        training = get_training_value(training)

        loss_name_prefix = f"{self.name}"

        if extra_prefix is not None:
            loss_name_prefix = f"{loss_name_prefix}_{extra_prefix}"

        inputs_shape = tf.shape(features)
        height = inputs_shape[-3]
        width = inputs_shape[-2]

        label = resize_image(label, (height, width), method="nearest")

        tf.debugging.check_numerics(features, "features contains nan or inf")

        flatten_features = flatten_hw(features)

        not_ignore_spatial_mask = tf.cast(label, tf.int32) != self.ignore_label  # [N, H, W, 1]
        not_ignore_spatial_mask = flatten_hw(not_ignore_spatial_mask)

        one_hot_label = get_flatten_one_hot_label(
            label, num_class=self.num_class, ignore_label=self.ignore_label
        )  # [N, HW, class]

        ####################################################################################

        class_sum_features, class_sum_non_zero_map = get_class_sum_features_and_counts(
            flatten_features, one_hot_label
        )  # [N, class, C]

        if self.use_batch_class_center:

            replica_context = tf.distribute.get_replica_context()

            class_sum_features_in_cross_batch = tf.reduce_sum(
                class_sum_features, axis=0, keepdims=True, name="class_sum_features_in_cross_batch"
            )
            class_sum_non_zero_map_in_cross_batch = tf.reduce_sum(
                class_sum_non_zero_map, axis=0, keepdims=True, name="class_sum_non_zero_map_in_cross_batch"
            )

            if replica_context:
                class_sum_features_in_cross_batch = replica_context.all_reduce(
                    tf.distribute.ReduceOp.SUM, class_sum_features_in_cross_batch
                )
                class_sum_non_zero_map_in_cross_batch = replica_context.all_reduce(
                    tf.distribute.ReduceOp.SUM, class_sum_non_zero_map_in_cross_batch
                )

            class_avg_features_in_cross_batch = tf.math.divide_no_nan(
                class_sum_features_in_cross_batch, class_sum_non_zero_map_in_cross_batch
            )  # [1, class, C]

            if self.use_last_class_center:

                batch_class_ignore_mask = tf.cast(class_sum_non_zero_map_in_cross_batch != 0, tf.int32)
                
                class_center_diff = class_avg_features_in_cross_batch - tf.cast(self.last_class_center, class_avg_features_in_cross_batch.dtype)
                class_center_diff *= (1 - self.last_class_center_decay) * tf.cast(batch_class_ignore_mask, class_center_diff.dtype)

                self.last_class_center.assign_add(class_center_diff)

                class_avg_features_in_cross_batch = tf.cast(self.last_class_center, tf.float32)

            class_avg_features = class_avg_features_in_cross_batch

        else:
            class_avg_features = tf.math.divide_no_nan(
                class_sum_features, class_sum_non_zero_map
            )  # [N, class, C]

        ####################################################################################

        if self.use_inter_class_loss and training:

            inter_class_relative_loss = 0

            if self.use_inter_c2c_loss:
                inter_class_relative_loss += get_inter_class_relative_loss(
                    class_avg_features, inter_c2c_loss_threshold=self.inter_c2c_loss_threshold,
                )

            if self.use_inter_c2p_loss:
                inter_class_relative_loss += get_pixel_inter_class_relative_loss(
                    flatten_features, class_avg_features, one_hot_label, inter_c2p_loss_threshold=self.inter_c2p_loss_threshold,
                )

            self.add_loss(inter_class_relative_loss * self.inter_class_loss_rate)
            self.add_metric(inter_class_relative_loss, name=f"{loss_name_prefix}_orl")

        if self.use_intra_class_loss:

            same_avg_value = tf.matmul(one_hot_label, class_avg_features)

            tf.debugging.check_numerics(same_avg_value, "same_avg_value contains nan or inf")

            self_absolute_loss = get_intra_class_absolute_loss(
                flatten_features,
                same_avg_value,
                remove_max_value=self.intra_class_loss_remove_max,
                not_ignore_spatial_mask=not_ignore_spatial_mask,
            )

            if training:
                self.add_loss(self_absolute_loss * self.intra_class_loss_rate)
                self.add_metric(self_absolute_loss, name=f"{loss_name_prefix}_sal")

            print("Using self-loss")

    def build(self, input_shape):

        # Note that, this is not the best design for specified architecture, but a trade-off for generalizability

        channels = input_shape[0][-1]
        channels = self.filters if channels > self.filters else channels

        print(f"car channels = {channels}")

        self.linear_conv = tf.keras.layers.Conv2D(channels, (1, 1), use_bias=True, name="linear_conv",)

        if self.apply_convs:
            self.end_conv = tf.keras.layers.Conv2D(channels, (1, 1), use_bias=False, name="end_conv",)
            self.end_norm = normalization(name="end_norm")

        if self.use_last_class_center:
            self.last_class_center = self.add_weight(
                name="last_class_center",
                shape=[1, self.num_class, channels],
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=False,
            )
            

    def call(self, inputs, training=None):

        inputs, label = inputs

        x = inputs

        # This linear conv (w/o norm&activation) can be merged 
        # to the next one (end_conv) during inference
        # Simple (x * w0 + b) * w1 dot product
        # We keep it for better understanding
        x = self.linear_conv(x) 

        y = tf.identity(x)

        if self.train_mode and get_training_value(training):

            x = tf.cast(x, tf.float32)

            tf.debugging.check_numerics(x, "inputs contains nan or inf")

            num_pooling_rates = len(self.pooling_rates)

            for i in range(num_pooling_rates):

                pooling_rate = self.pooling_rates[i]

                sub_x = tf.identity(x, name=f"x_in_rate_{pooling_rate}")

                if pooling_rate > 1:
                    stride_size = (1, pooling_rate, pooling_rate, 1)
                    sub_x = tf.nn.avg_pool2d(sub_x, stride_size, stride_size, padding="SAME")

                self.add_car_losses(sub_x, label=label, extra_prefix=str(pooling_rate), training=training)

        if self.apply_convs:
            y = self.end_conv(y)
            y = self.end_norm(y, training=training)
            y = tf.nn.relu(y)

        return y
