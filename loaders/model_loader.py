import iseg.static_strings as ss

import iseg.layers.normalizations as norm
import iseg.utils.common

from absl import flags
from common_flags import FLAGS
from iseg.modelhelper import model_common_setup

from carnet import ClassAwareRegularizationNetwork


# Pretrained weights path

flags.DEFINE_string("resnet50_weights_path", None, "H5 weights for Resnet50")
flags.DEFINE_string("resnet101_weights_path", None, "H5 weights for Resnet101")
flags.DEFINE_string("resnet152_weights_path", None, "H5 weights for Resnet152")

flags.DEFINE_string("xception65_weights_path", None, "H5 weights for Xception65")
flags.DEFINE_string("mobilenetv2_weights_path", None, "H5 weights for MobileNetV2")

flags.DEFINE_string("efficientnetb0_weights_path", None, "H5 weights for efficientnetb0")
flags.DEFINE_string("efficientnetb1_weights_path", None, "H5 weights for efficientnetb1")
flags.DEFINE_string("efficientnetb2_weights_path", None, "H5 weights for efficientnetb2")
flags.DEFINE_string("efficientnetb3_weights_path", None, "H5 weights for efficientnetb3")
flags.DEFINE_string("efficientnetb4_weights_path", None, "H5 weights for efficientnetb4")
flags.DEFINE_string("efficientnetb5_weights_path", None, "H5 weights for efficientnetb5")
flags.DEFINE_string("efficientnetb6_weights_path", None, "H5 weights for efficientnetb6")
flags.DEFINE_string("efficientnetb7_weights_path", None, "H5 weights for efficientnetb7")

flags.DEFINE_string("swin_tiny_224_weights_path", None, "H5 weights for Swin tiny 224")
flags.DEFINE_string("swin_base_384_weights_path", None, "H5 weights for Swin base 384")
flags.DEFINE_string("swin_large_384_weights_path", None, "H5 weights for Swin large 384")

flags.DEFINE_string("hrnet_w48_weights_path", None, "H5 weights for HRNet-W48")
flags.DEFINE_string("hrnet_w32_weights_path", None, "H5 weights for HRNet-W32")

flags.DEFINE_string("convnext_tiny_weights_path", None, "H5 weights for ConvNext-Tiny")
flags.DEFINE_string("convnext_large_weights_path", None, "H5 weights for ConvNext-Large")
flags.DEFINE_string("convnext_xlarge_weights_path", None, "H5 weights for ConvNext-XLarge")

flags.DEFINE_enum(
    "global_norm_method",
    norm.BATCH_NORM,
    [norm.BATCH_NORM, norm.SYNC_BATCH_NORM, norm.GROUP_NROM],
    "Global norm method",
)
flags.DEFINE_enum("global_resize_method", "bilinear", ["bilinear", "nearest"], "global resize method")

flags.DEFINE_string("backbone", ss.RESNET50, "name of the backbone")
flags.DEFINE_integer("output_stride", 32, "output stride")
flags.DEFINE_boolean("aux_loss", False, "Apply aux loss")
flags.DEFINE_float("aux_loss_rate", 0.2, "rate of aux loss")

flags.DEFINE_float("weight_decay", None, "Weight decay")
flags.DEFINE_bool("decay_norm_vars", False, "Weight decay on normliaztions")
flags.DEFINE_float("bn_momentum", None, "Batch norm momentum")
flags.DEFINE_float("bn_epsilon", None, "Batch norm epsilon")

flags.DEFINE_float("backbone_bn_momentum", None, "Backbone batch norm momentum")

# CAR Settings

flags.DEFINE_bool("use_intra_class_loss", False, "use self loss in head")
flags.DEFINE_bool("use_inter_class_loss", False, "use other loss in head")
flags.DEFINE_bool("intra_class_loss_remove_max", False, "self loss remove max in head")
flags.DEFINE_bool("use_inter_c2p_loss", False, "other loss use other pixel")
flags.DEFINE_float("intra_class_loss_rate", 1, "Expand ratio for other loss")
flags.DEFINE_float("inter_class_loss_rate", 1, "Expand ratio for other loss")
flags.DEFINE_string("head", "nl", "head of label attention")
flags.DEFINE_bool("apply_car", False, "apply car on head")
flags.DEFINE_bool("apply_car_convs", False, "apply convs to replace ")
flags.DEFINE_bool("use_batch_class_center", True, "use_batch_class_center")
flags.DEFINE_bool("use_last_class_center", False, "use_last_class_center")
flags.DEFINE_float("last_class_center_decay", 0.9, "last_class_center_decay")
flags.DEFINE_float("inter_c2c_loss_threshold", 0.50, "inter_c2c_loss_threshold")
flags.DEFINE_float("inter_c2p_loss_threshold", 0.25, "inter_c2p_loss_threshold")
flags.DEFINE_bool("use_multi_lr", False, "car use multiple lrs during training")


def get_sliding_window_crop_size():

    sliding_window_crop_size = None

    if (FLAGS.sliding_window_crop_height is not None) and (FLAGS.sliding_window_crop_width is not None):

        sliding_window_crop_size = (FLAGS.sliding_window_crop_height, FLAGS.sliding_window_crop_width)

    return sliding_window_crop_size


def load_model(distribute_strategy, num_class, ignore_label=0):

    norm.global_norm_method = lambda: FLAGS.global_norm_method
    iseg.utils.common.DEFAULT_IMAGE_RESIZE_METHOD = FLAGS.global_resize_method

    model = None
    
    with distribute_strategy.scope():
        
        model = ClassAwareRegularizationNetwork(
            backbone_name=FLAGS.backbone,
            backbone_weights_path=__get_weight_path(),
            head_name=FLAGS.head,
            apply_car=FLAGS.apply_car,
            apply_car_convs=FLAGS.apply_car_convs,
            num_class=num_class,
            ignore_label=ignore_label,
            output_stride=FLAGS.output_stride,
            use_aux_loss=FLAGS.aux_loss,
            aux_loss_rate=FLAGS.aux_loss_rate,
            train_mode=FLAGS.mode == ss.TRAIN,
            use_intra_class_loss=FLAGS.use_intra_class_loss,
            use_inter_class_loss=FLAGS.use_inter_class_loss,
            use_inter_c2p_loss=FLAGS.use_inter_c2p_loss,
            intra_class_loss_remove_max=FLAGS.intra_class_loss_remove_max,
            intra_class_loss_rate=FLAGS.intra_class_loss_rate,
            inter_class_loss_rate=FLAGS.inter_class_loss_rate,
            use_batch_class_center=FLAGS.use_batch_class_center,
            use_last_class_center=FLAGS.use_last_class_center,
            last_class_center_decay=FLAGS.last_class_center_decay,
            inter_c2c_loss_threshold=FLAGS.inter_c2c_loss_threshold,
            inter_c2p_loss_threshold=FLAGS.inter_c2p_loss_threshold,
            use_multi_lr=FLAGS.use_multi_lr,
        )

    return model_common_setup(
        model=model,
        restore_checkpoint=FLAGS.restore_checkpoint,
        checkpoint_dir=FLAGS.checkpoint_dir,
        max_checkpoints_to_keep=FLAGS.max_checkpoints_to_keep,
        weight_decay=FLAGS.weight_decay,
        decay_norm_vars=FLAGS.decay_norm_vars,
        bn_epsilon=FLAGS.bn_epsilon,
        bn_momentum=FLAGS.bn_momentum,
        backbone_bn_momentum=FLAGS.backbone_bn_momentum,
        inference_sliding_window_size=get_sliding_window_crop_size(),
    )


def __get_weight_path():

    weights_path_dict = {
        ss.RESNET50: FLAGS.resnet50_weights_path,
        ss.RESNET52: FLAGS.resnet50_weights_path,
        ss.RESNET101: FLAGS.resnet101_weights_path,
        ss.RESNET103: FLAGS.resnet101_weights_path,
        ss.RESNET152: FLAGS.resnet152_weights_path,
        ss.XCEPTION65: FLAGS.xception65_weights_path,
        ss.EFFICIENTNETB0: FLAGS.efficientnetb0_weights_path,
        ss.EFFICIENTNETB1: FLAGS.efficientnetb1_weights_path,
        ss.EFFICIENTNETB2: FLAGS.efficientnetb2_weights_path,
        ss.EFFICIENTNETB3: FLAGS.efficientnetb3_weights_path,
        ss.EFFICIENTNETB4: FLAGS.efficientnetb4_weights_path,
        ss.EFFICIENTNETB5: FLAGS.efficientnetb5_weights_path,
        ss.EFFICIENTNETB6: FLAGS.efficientnetb6_weights_path,
        ss.EFFICIENTNETB7: FLAGS.efficientnetb7_weights_path,
        ss.MOBILENETV2: FLAGS.mobilenetv2_weights_path,
        ss.SWIN_TINY_224: FLAGS.swin_tiny_224_weights_path,
        ss.SWIN_BASE_384: FLAGS.swin_base_384_weights_path,
        ss.SWIN_LARGE_384: FLAGS.swin_large_384_weights_path,
        ss.HRNET_W48 : FLAGS.hrnet_w48_weights_path,
        ss.HRNET_W32 : FLAGS.hrnet_w32_weights_path,
        ss.CONVNEXT_TINY : FLAGS.convnext_tiny_weights_path,
        ss.CONVNEXT_LARGE: FLAGS.convnext_large_weights_path,
        ss.CONVNEXT_XLARGE: FLAGS.convnext_xlarge_weights_path,
        ss.PLACEHOLDER: None,
    }

    return weights_path_dict[FLAGS.backbone]