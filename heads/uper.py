import tensorflow as tf

from iseg.layers.poolings import adaptive_average_pooling_2d
from iseg.layers.fpn import FeaturePyramidNetwork
from iseg.layers.model_builder import ConvBnRelu
from iseg.utils.common import resize_image
from iseg.vis.vismanager import get_visualization_manager

from heads.baseline import Baseline

# This code is manually translated from mmsegmentaion repo
# https://github.com/open-mmlab/mmsegmentation
# We also ref to the offical repo
# https://github.com/CSAILVision/unifiedparsing


class UperNetHead(Baseline):
    def __init__(
        self, use_aux_loss=False, train_mode=False, baseline_mode=True, replace_2nd_last_conv=False, name=None
    ):

        super().__init__(
            train_mode=train_mode,
            baseline_mode=baseline_mode,
            replace_2nd_last_conv=replace_2nd_last_conv,
            car_pooling_rates=[2],
            name=name,
        )

        self.use_aux_loss = use_aux_loss
        self.train_mode = train_mode

        self.psp_modules = PyramidPoollingModule(filters=512, pool_sizes=[1, 2, 3, 6], name="ppm")

        self.bottleneck = ConvBnRelu(512, (3, 3), name="bottleneck")

        self.fpn = FeaturePyramidNetwork(skip_conv_filters=512, name="fpn")
        self.fpn_convs = UperConvsBlocks()

        if not self.replace_2nd_last_conv:
            self.fpn_bottleneck = ConvBnRelu(512, (3, 3), name="fpn_bottleneck")
        elif self.baseline_mode and self.replace_2nd_last_conv:
            self.fpn_bottleneck = ConvBnRelu(512, (1, 1), name="fpn_bottleneck")

        self.vis_manager = get_visualization_manager()
        

    def call(self, inputs, training=None):

        label = None

        if self.train_mode:
            endpoints, label = inputs
        else:
            endpoints = inputs

        endpoints: list = endpoints
        endpoints = endpoints[1:]

        if self.train_mode:
            label = tf.cast(label, endpoints[0].dtype)
            label = tf.expand_dims(label, axis=-1)  # [N, H, W, 1]
            label = resize_image(label, size=tf.shape(endpoints[0])[1:3], method="nearest")  # [N, H, W, 1]

        rich_feature = endpoints[-1]
        rich_feature = self.psp_modules(rich_feature, training=training)
        rich_feature = self.bottleneck(rich_feature, training=training)

        endpoints[-1] = rich_feature

        endpoints = self.fpn(endpoints, training=training)[:-1]  # remove last rich feature
        endpoints = self.fpn_convs(endpoints, training=training)

        endpoints.append(rich_feature)

        for i in range(len(endpoints)):
            endpoint = endpoints[i]

            if i > 0:
                endpoint = resize_image(endpoint, size=tf.shape(endpoints[0])[1:3])

            endpoints[i] = endpoint

        y = tf.concat(endpoints, axis=-1)

        if not self.baseline_mode:
            y = self.gta([y, label], training=training)

        if not self.replace_2nd_last_conv or (self.baseline_mode and self.replace_2nd_last_conv):
            y = self.fpn_bottleneck(y, training=training)

        if self.vis_manager.recording:
            self.vis_manager.easy_add(y, name="gta_rich")

        return y


class UperConvsBlocks(tf.keras.Model):
    def __init__(self, k_size=(3, 3), name=None):
        super().__init__(name=name)

        self.k_size = k_size

    def build(self, input_shape):

        num_inputs = len(input_shape)

        self.convs = [ConvBnRelu(input_shape[i][-1], self.k_size, name=f"fpn_conv_{i}") for i in range(num_inputs)]

    def call(self, inputs, training=None):

        endpoints = inputs

        assert len(self.convs) == len(endpoints)

        for i in range(len(endpoints)):
            endpoints[i] = self.convs[i](endpoints[i], training=training)

        return endpoints


class PyramidPoollingModule(tf.keras.Model):
    def __init__(self, filters=256, pool_sizes=[1], name=None):
        super().__init__(name=name)

        assert len(pool_sizes) > 0, "Pool sizes len cannot be 0 !"

        self.pool_sizes = pool_sizes
        self.conv_blocks = [
            ConvBnRelu(filters=filters, kernel_size=3, name=f"{self.name}/conv_size_{size}") for size in pool_sizes
        ]

    def call(self, inputs, training=None):

        results = [inputs]
        inputs_size = tf.shape(inputs)[1:3]

        for i in range(len(self.pool_sizes)):
            x = adaptive_average_pooling_2d(inputs, size=self.pool_sizes[i])
            x = self.conv_blocks[i](x, training=training)

            x = resize_image(x, inputs_size)
            results.append(x)

        return tf.concat(results, axis=-1, name="concat_result")
