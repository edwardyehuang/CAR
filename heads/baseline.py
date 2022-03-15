# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from car_core.car import ClassAwareRegularization


class Baseline(tf.keras.Model):
    def __init__(
        self,
        train_mode=False,
        baseline_mode=True,
        replace_2nd_last_conv=False,
        car_pooling_rates=[1],
        car_filters=512,
        name=None,
    ):

        super().__init__(name=name)

        self.train_mode = train_mode
        self.baseline_mode = baseline_mode
        self.replace_2nd_last_conv = replace_2nd_last_conv

        print(f"------Baseline settings------")
        print(f"------train_mode = {train_mode}")
        print(f"------baseline_mode = {baseline_mode}")
        print(f"------replace_2nd_last_conv = {replace_2nd_last_conv}")
        print()

        if not baseline_mode:

            # gta is the project initial name for CAR, gt stands for Ground Truth
            # We keep it for backward compatibility (ckpts etc.)

            self.gta = ClassAwareRegularization(pooling_rates=car_pooling_rates, filters=car_filters, name="g")
