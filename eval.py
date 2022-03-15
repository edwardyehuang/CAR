import loaders.dataset_loader as dataset_loader
import loaders.model_loader as model_loader

from absl import app
from common_flags import FLAGS

from iseg.core_env import common_env_setup
from iseg.evaluation import evaluate


def main(argv):

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

    _, val_ds, num_class, ignore_label, _, _, _, val_image_count = dataset_loader.load_dataset_from_flags()

    model_helper = model_loader.load_model(strategy, num_class, ignore_label=ignore_label)

    evaluate(
        strategy,
        model_helper.model,
        val_ds,
        FLAGS.gpu_batch_size,
        num_class,
        ignore_label,
        scale_rates=FLAGS.scale_rates,
        flip=FLAGS.flip,
        val_image_count=val_image_count,
    )


    if FLAGS.press_key_to_end:
        input("Press any key to exit")


if __name__ == "__main__":
    app.run(main)
