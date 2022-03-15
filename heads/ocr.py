import tensorflow as tf

from iseg.layers.model_builder import ConvBnRelu, get_tensor_shape
from iseg.utils.attention_utils import flatten_hw
from iseg.utils.common import resize_image
from iseg.vis.vismanager import get_visualization_manager
from heads.baseline import Baseline


class ObjectContextualRepresentationsHead(Baseline):
    def __init__(self, 
        num_class=21, 
        use_aux_loss=True, 
        train_mode=False,
        baseline_mode=True,
        replace_2nd_last_conv=False,
        name=None):

        super().__init__(
            train_mode=train_mode, 
            baseline_mode=baseline_mode, 
            replace_2nd_last_conv=replace_2nd_last_conv, name=name,
            car_pooling_rates=[2],
            car_filters=512,
        )

        self.use_aux_loss = use_aux_loss
        self.num_class = num_class

        self.down_conv = ConvBnRelu(512, (3, 3), name="down_conv")

        self.spatial_gather_module = SpatialGatherModule()
        self.object_attention = ObjectAttentionBlock(filters=256)

        if not self.replace_2nd_last_conv:
            self.end_conv = ConvBnRelu(512, name="end_conv")

        self.vis_manager = get_visualization_manager()


    def build (self, input_shape):

        if self.train_mode:
            input_shape = input_shape[0]
        
        channels = input_shape[-1][-1]

        self.dsn_conv = ConvBnRelu(channels, (3, 3), name="dsn_conv")

        self.dsn_logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name="dsn_logits_conv")
        self.logtis_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name="main_logits_conv")


    def call(self, inputs, training=None):

        label = None

        if self.train_mode:
            endpoints, label = inputs
        else:
            endpoints = inputs


        # https://github.com/NVIDIA/semantic-segmentation/blob/main/network/ocrnet.py
        # https://github.com/openseg-group/openseg.pytorch/issues/56

        x_aux = endpoints[-1]
        x = endpoints[-1]

        if self.train_mode:
            label = tf.cast(label, x.dtype)
            label = tf.expand_dims(label, axis=-1)  # [N, H, W, 1]
            label = resize_image(label, size=tf.shape(x)[1:3], method="nearest")  # [N, H, W, 1]

        x_aux = self.dsn_conv(x_aux, training=training)
        logits_aux = self.dsn_logits_conv(x_aux)  # [N, H, W, class]
        logits_aux = tf.cast(logits_aux, tf.float32)

        x = self.down_conv(x, training=training)
        x_pre_ocr = tf.identity(x, name="x_pre_ocr")

        class_center = self.spatial_gather_module([x, logits_aux], training=training)  # [N, class, 512]

        x = self.object_attention([x, class_center], training=training)  # [N, H, W, C]
        x = tf.concat([x, x_pre_ocr], axis=-1)  # [N, H, W, C * 2]

        if not self.baseline_mode:
            x = self.gta((x, label), training=training)

        if not self.replace_2nd_last_conv:
            x = self.end_conv(x, training=training)

        if self.vis_manager.recording:
            self.vis_manager.easy_add(x, name="gta_rich")

        logits = self.logtis_conv(x)

        if self.use_aux_loss:
            logits = [logits, logits_aux]

        return logits


class SpatialGatherModule(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, inputs, training=None):

        x, prob = inputs  # x : [N, H, W, C] and prob : [N, H, W, class]

        x = flatten_hw(x)  # [N, HW, C]

        prob = flatten_hw(prob)  # [N, HW, class]
        prob = tf.transpose(prob, [0, 2, 1])  # [N, class, HW]
        prob = tf.nn.softmax(prob)  # [N, class, HW] Spatial softmax

        ocr_context = tf.matmul(prob, x)  # [N, C, class]

        return tf.transpose(ocr_context, [0, 2, 1])  # [N, class, C]


class ObjectAttentionBlock(tf.keras.Model):
    def __init__(self, filters, name=None):

        super().__init__(name=name)

        self.pixel_query_conv0 = ConvBnRelu(filters=filters, name="pixel_query_conv0")
        self.pixel_query_conv1 = ConvBnRelu(filters=filters, name="pixel_query_conv1")

        self.class_key_conv0 = ConvBnRelu(filters=filters, name="class_key_conv0")
        self.class_key_conv1 = ConvBnRelu(filters=filters, name="class_key_conv1")

        self.value_conv = ConvBnRelu(filters=filters, name="value_conv")

    def build(self, input_shape):

        channels = input_shape[0][-1]

        self.up_conv = ConvBnRelu(filters=channels, name="up_conv")

    def call(self, inputs, training=None):

        x, class_center = inputs

        batch_size, height, width, _ = get_tensor_shape(x)

        class_center = tf.expand_dims(class_center, axis=1)  # [N, 1, class, C]

        query = self.pixel_query_conv0(x, training=training)
        query = self.pixel_query_conv1(query, training=training)  # [N, H, W, C]
        query = tf.reshape(query, [batch_size, height * width, query.shape[-1]]) # [N, HW, C]

        key = self.class_key_conv0(class_center, training=training)
        key = self.class_key_conv1(key, training=training)
        key = tf.squeeze(key, axis=1)  # [N, class, C]
        key = tf.transpose(key, [0, 2, 1])  # [N, C, class]

        value = self.value_conv(class_center, training=training)
        value = tf.squeeze(value, axis=1)  # [N, class, C]

        pixel_class_sim = tf.matmul(query, key)  # [N, HW, class]
        pixel_class_sim /= tf.sqrt(tf.cast(query.shape[-1], pixel_class_sim.dtype))
        pixel_class_sim = tf.nn.softmax(pixel_class_sim)  # [N, HW, class]

        y = tf.matmul(pixel_class_sim, value)  # [N, HW, C]
        y = tf.reshape(y, [batch_size, height, width, y.shape[-1]])  # [N, H, W, C]

        y = self.up_conv(y, training=training)  # 256->512

        return y
