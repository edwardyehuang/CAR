# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
from iseg.layers.self_attention import flatten_hw


def square_mean_loss(loss, inverse_order=True):

    if inverse_order:
        return tf.square(tf.reduce_mean(loss))
    else:
        return tf.reduce_mean(tf.square(loss))
        

def get_flatten_one_hot_label(label, num_class=21, ignore_label=0):

    label = flatten_hw(label)
    label = tf.cast(label, tf.int32)  # [N, HW, 1]
    label = tf.squeeze(label, axis=-1)

    if ignore_label == 0:
        label -= 1
        tf.assert_greater(num_class, tf.reduce_max(label))

    one_hot_label = tf.one_hot(label, num_class)  # [N, HW, class]

    return one_hot_label


def get_class_sum_features_and_counts(features, one_hot_label):

    # features [N, HW, C]
    # label [N, HW, class] int

    class_mask = tf.transpose(one_hot_label, [0, 2, 1])  # [N, class, HW]

    non_zero_map = tf.math.count_nonzero(class_mask, axis=-1, keepdims=True, dtype=tf.int32)
    non_zero_map = tf.cast(non_zero_map, dtype=features.dtype)  # [N, class, 1]

    class_mask = tf.cast(class_mask, dtype=features.dtype)

    class_sum_feature = tf.matmul(class_mask, features)  # [N, class, C]

    return class_sum_feature, non_zero_map


def get_inter_class_relations(
    query, 
    key=None, 
    apply_scale=True,
    ):

    if key is None:
        key = tf.identity(query)

    key = tf.transpose(key, [0, 2, 1])  # [N, C, class]
    key = tf.stop_gradient(key)

    attention = tf.matmul(query, key)  # [N, class, class]

    num_class = tf.shape(key)[-1]
    diag = get_class_diag(num_class, query.dtype)


    if apply_scale:
        attention_scale = tf.sqrt(tf.cast(query.shape[-1], query.dtype))
        attention /= attention_scale

    attention = tf.nn.softmax(attention)

    return attention, diag


def get_class_diag(num_class, dtype=tf.float32):

    ones = tf.ones(tf.expand_dims(num_class, axis=-1), dtype=tf.int32)
    diag = tf.linalg.diag(ones)
    diag = tf.cast(diag, dtype=dtype)  # [N, class, class]

    return diag


def get_inter_class_relative_loss(
    class_features_query, 
    class_features_key=None, 
    inter_c2c_loss_threshold=0.5):

    # class_features [N, class, C]

    class_relation, diag = get_inter_class_relations(
        class_features_query, class_features_key,
    )  # [N, class, class]

    num_class = tf.shape(class_relation)[-1]

    other_relation = class_relation * (1 - diag)  # [N, class, class]

    threshold = inter_c2c_loss_threshold / tf.cast(num_class - 1, class_relation.dtype)

    other_relation = tf.where(other_relation > threshold, other_relation - threshold, 0)

    loss = tf.reduce_sum(other_relation, axis=-1)  # [N, class]

    loss = tf.clip_by_value(
        loss, clip_value_min=tf.keras.backend.epsilon(), clip_value_max=1 - tf.keras.backend.epsilon(), name="loss_clip"
    )

    loss = square_mean_loss(loss)

    return loss


def get_intra_class_absolute_loss(x, avg_value, remove_max_value=False, not_ignore_spatial_mask=None):

    avg_value = tf.stop_gradient(avg_value)
    value_diff = tf.math.abs(avg_value - x)  # [N, HW, C]

    if not_ignore_spatial_mask is not None:
        value_diff *= tf.cast(not_ignore_spatial_mask, value_diff.dtype)

    value_diff = tf.transpose(value_diff, [0, 2, 1])  # [N, C, HW]
    assert len(value_diff.shape) == 3, "ndim must be 3"

    if remove_max_value:
        value_diff = tf.sort(value_diff, direction="ASCENDING")  # [N, C, HW]
        threshold = 1  # tf.cast(tf.cast(tf.shape(value_diff)[-1], tf.float32) * 0.8, tf.int32)
        value_diff = value_diff[:, :, :-threshold]  # [N, C, HW - 1]

    loss = value_diff
    loss = square_mean_loss(loss)
    loss = tf.clip_by_value(loss, 1e-5, loss)

    return loss


def get_pixel_inter_class_relative_loss(x, class_avg_feature, one_hot_label, inter_c2p_loss_threshold=0.25):

    # x : [N, HW, C]
    # class_avg_feature : # [N, class, C]
    # one_hot_label : [N, HW, class]

    class_avg_feature = tf.stop_gradient(class_avg_feature)
    class_avg_feature = tf.transpose(class_avg_feature, [0, 2, 1])  # [N, C, class]

    energy = tf.matmul(x, class_avg_feature)  # [N, HW, class]

    self_energy = class_avg_feature * class_avg_feature  # [N, C, class]
    self_energy = tf.reduce_sum(self_energy, axis=1, keepdims=True)  # [N, 1, class]

    other_label_mask = tf.cast(1 - tf.cast(one_hot_label, tf.int32), energy.dtype)


    energy *= other_label_mask  # [N, HW, class]
    energy += self_energy * one_hot_label

    energy_scale = tf.sqrt(tf.cast(x.shape[-1], x.dtype))
    energy /= energy_scale
    inter_c2p_relation = tf.nn.softmax(energy, axis=-1)  # [N, HW, class]

    num_class = tf.shape(inter_c2p_relation)[-1]
    num_class = tf.cast(num_class, inter_c2p_relation.dtype)

    threshold = inter_c2p_loss_threshold / (num_class - 1)

    other_c2p_relation = inter_c2p_relation * other_label_mask  # [N, HW, class]
    other_c2p_relation = tf.where(other_c2p_relation > threshold, other_c2p_relation - threshold, 0)
    other_c2p_relation = tf.reduce_sum(other_c2p_relation, axis=-1)  # [N, HW]

    other_c2p_relation = tf.clip_by_value(other_c2p_relation, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    loss = other_c2p_relation
    loss = square_mean_loss(loss)

    return loss
