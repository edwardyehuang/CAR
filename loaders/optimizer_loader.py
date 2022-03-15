from absl import flags

from iseg.core_optimizer import get_optimizer

from common_flags import FLAGS


flags.DEFINE_float("initial_lr", 7e-3, "Initial learning rate")
flags.DEFINE_float("end_lr", 0, "End learning rate")
flags.DEFINE_enum("decay_strategy", "none", ["none", "poly", "cosine"], "Decay strategy")
flags.DEFINE_enum("optimizer", "sgd", ["sgd", "amsgrad", "adam", "adamw"], "optimizer")

flags.DEFINE_multi_float("lr_ratio", [1.0], "Ratio for multi learning rate")
flags.DEFINE_float("adamw_weight_decay", 0.0, "AdamW weihts decay")


def load_optimizer_from_flags():

    lr_ratio = FLAGS.lr_ratio
    initial_lr = FLAGS.initial_lr

    if len(lr_ratio) > 1:
        initial_lr = [initial_lr * lr_ratio[i] for i in range(len(lr_ratio))]

    optimizer = get_optimizer(
        initial_lr=initial_lr,
        end_lr=FLAGS.end_lr,
        epoch_steps=FLAGS.epoch_steps,
        train_epoch=FLAGS.train_epoch,
        decay_strategy=FLAGS.decay_strategy,
        optimizer=FLAGS.optimizer,
        sgd_momentum_rate=FLAGS.sgd_momentum_rate,
        adamw_weight_decay=FLAGS.adamw_weight_decay,
    )

    if isinstance(optimizer, list):
        print("Found multiple optimizers")

    return optimizer