import loaders.dataset_loader as dataset_loader
import loaders.model_loader as model_loader
import loaders.optimizer_loader as optimizer_loader

from absl import app
from common_flags import FLAGS

from iseg.core_env import common_env_setup
from iseg.core_train import CoreTrain


def train(argv):

    # tf.debugging.disable_traceback_filtering()

    strategy = common_env_setup(
        run_eagerly=False,
        gpu_memory_growth=FLAGS.gpu_memory_growth,
        cuda_visible_devices=FLAGS.cuda_visible_devices,
        tpu_name=FLAGS.tpu_name,
        random_seed=FLAGS.random_seed,
        mixed_precision=FLAGS.mixed_precision,
        use_deterministic=True,
    )

    (
        train_ds,
        val_ds,
        num_class,
        ignore_label,
        class_weights,
        train_size,
        val_size,
        val_image_count,
    ) = dataset_loader.load_dataset_from_flags()

    model_helper = model_loader.load_model(strategy, num_class, ignore_label=ignore_label)

    model_helper.set_optimizer(optimizer_loader.load_optimizer_from_flags())

    training = CoreTrain(
        model_helper, train_ds, val_ds, val_image_count=val_image_count, use_tpu=FLAGS.tpu_name is not None
    )

    training.train(
        distribute_strategy=strategy,
        num_class=num_class,
        ignore_label=ignore_label,
        class_weights=class_weights,
        batch_size=FLAGS.gpu_batch_size,
        eval_batch_size=FLAGS.eval_gpu_batch_size,
        shuffle_rate=FLAGS.shuffle,
        epoch_steps=FLAGS.epoch_steps,
        initial_epoch=FLAGS.initial_epoch,
        train_epoches=FLAGS.train_epoch,
        tensorboard_dir=FLAGS.tensorboard_dir,
        verbose=1 if FLAGS.training_progress_bar else 2,
    )


    if FLAGS.press_key_to_end:
        input("Press any key to exit")


if __name__ == "__main__":
    app.run(train)
