# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# CarNet is not a standalone network
# Instead, it provides a wrapper to test
# car on multiple baseline networks

import functools
import tensorflow as tf
import iseg.static_strings as ss

import car_core.car as car

from iseg.layers.core_model_ext import SegManaged

NL = "nl"
UPER = "uper"
FCN = "fcn"
DEEPLAB = "deeplab"
CCNET = "ccnet"
DANET = "danet"
CPNET = "cpnet"
OCR   = "ocr"
CAA   = "caa"


def ClassAwareRegularizationNetwork(
    backbone_name=ss.RESNET101,
    backbone_weights_path=None,
    head_name="nl",
    num_class=21,
    ignore_label=0,
    output_stride=8,
    use_aux_loss=False,
    aux_loss_rate=0.4,
    train_mode=True,
    apply_car=False,
    apply_car_convs=False,
    use_intra_class_loss=True,
    use_inter_class_loss=True,
    intra_class_loss_remove_max=False,
    use_inter_c2p_loss=False,
    intra_class_loss_rate=1,
    inter_class_loss_rate=1,
    use_batch_class_center=True,
    use_last_class_center=False,
    last_class_center_decay=0.9,
    inter_c2c_loss_threshold=0.5,
    inter_c2p_loss_threshold=0.25,
    use_multi_lr=False,
    name=None,
):

    heads_num_aux_loss = {
        NL:1,
        UPER:0,
        FCN:0,
        DEEPLAB:0,
        CCNET:1,
        DANET:2,
        CPNET:1,
        OCR: 1,
        CAA: 1,
    }

    head_name_lower = head_name.lower()

    print(f"------General settings------")
    print(f"------head_name = {head_name}")
    print(f"------apply_car = {apply_car}")
    print(f"------apply_car_convs = {apply_car_convs}")
    print(f"------use_multi_lr = {use_multi_lr}")
    print(f"------use_aux_loss = {use_aux_loss}")
    print(f"------aux_loss_rate = {aux_loss_rate}")
    print()

    general_kwargs = {
        "use_aux_loss": use_aux_loss,
    }

    car_kwargs = {
        "train_mode": train_mode,
        "use_intra_class_loss": use_intra_class_loss,
        "use_inter_class_loss": use_inter_class_loss,
        "intra_class_loss_remove_max": intra_class_loss_remove_max,
        "use_inter_c2p_loss": use_inter_c2p_loss,
        "intra_class_loss_rate": intra_class_loss_rate,
        "inter_class_loss_rate": inter_class_loss_rate,
        "use_batch_class_center": use_batch_class_center,
        "use_last_class_center": use_last_class_center,
        "last_class_center_decay": last_class_center_decay,
        "num_class": num_class,
        "ignore_label": ignore_label,
        "inter_c2c_loss_threshold": inter_c2c_loss_threshold,
        "inter_c2p_loss_threshold": inter_c2p_loss_threshold,
        "apply_convs": apply_car_convs,
    }

    head_kwargs = {
        "train_mode": train_mode and apply_car,
        "baseline_mode": not apply_car,
        "replace_2nd_last_conv": apply_car_convs,
    }

    car.ClassAwareRegularization = functools.partial(car.ClassAwareRegularization, **car_kwargs)

    use_custom_logits = False
    custom_aux_loss_fns = []
    logits_upsample_masks = None


    if head_name_lower == NL:
        from heads.nonlocals import LabelNonLocalHead

        head = LabelNonLocalHead(**general_kwargs, **head_kwargs)
    elif head_name_lower == UPER:
        from heads.uper import UperNetHead

        head = UperNetHead(**general_kwargs, **head_kwargs)
    elif head_name_lower == FCN:
        from heads.fcn import FCNHead

        head = FCNHead(**general_kwargs, **head_kwargs)
    elif head_name_lower == DEEPLAB:
        from heads.deeplabv3 import DeepLabv3Head

        head = DeepLabv3Head(output_stride=output_stride, **general_kwargs, **head_kwargs,)
    elif head_name_lower == CCNET:
        from heads.ccnet import CrissCrossAttentionHead

        head = CrissCrossAttentionHead(**general_kwargs, **head_kwargs)
    elif head_name_lower == DANET:
        from heads.danet import DualAttentionHead

        head = DualAttentionHead(**general_kwargs, **head_kwargs)
    elif head_name_lower == CPNET:
        from heads.cpnet import ContextPriorHead, affinity_loss

        prior_size = (32 // output_stride) * 16

        use_custom_logits = True
        custom_aux_loss_fns = [affinity_loss]
        logits_upsample_masks = [True, False] if train_mode else None

        head = ContextPriorHead(prior_size=prior_size, num_class=num_class, **general_kwargs, **head_kwargs)
    elif head_name_lower == OCR:
        from heads.ocr import ObjectContextualRepresentationsHead

        use_custom_logits = True

        head = ObjectContextualRepresentationsHead(num_class=num_class, **general_kwargs, **head_kwargs)
    elif head_name_lower == CAA:
        from heads.caa import ChannelizedAxialAttentionHead

        use_jpu = output_stride == 32

        head = ChannelizedAxialAttentionHead(use_jpu=use_jpu, **general_kwargs, **head_kwargs)
    else:
        raise ValueError(f"Currently not supported {head_name}")

    model = SegManaged(
        backbone_name=backbone_name,
        backbone_weights_path=backbone_weights_path,
        output_stride=output_stride,
        num_class=num_class,
        num_aux_loss=0 if not use_aux_loss else heads_num_aux_loss[head_name_lower],
        aux_loss_rate=aux_loss_rate,
        label_as_inputs=train_mode,
        label_as_backbone_inputs=False,
        label_as_head_inputs=apply_car,
        use_custom_logits=use_custom_logits,
        custom_aux_loss_fns=custom_aux_loss_fns,
        logits_upsample_masks=logits_upsample_masks,
        name=name,
    )

    model.head = head

    if train_mode and use_multi_lr:
        model((tf.ones([1, 513, 513, 3]), tf.ones([1, 513, 513])), training=False)
        model.layers_for_multi_optimizers = [model.backbone, model.head]

    return model