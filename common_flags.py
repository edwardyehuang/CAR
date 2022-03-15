import iseg.static_strings as ss

from absl import flags
FLAGS = flags.FLAGS

# System settings
flags.DEFINE_string("cuda_visible_devices", None, "visible cuda devices")

# Common settings

flags.DEFINE_enum("mode", ss.TRAIN, [ss.TRAIN, ss.VAL, ss.TEST_DIR], "mode")
flags.DEFINE_string("tensorboard_dir", None, "Path of tensorboard dir")
flags.DEFINE_string("checkpoint_dir", None, "Path of dir where the checkpoints are stored")
flags.DEFINE_string("visualize_output_dir", None, "Path of the dir to output the visualize results")

flags.DEFINE_boolean("gpu_memory_growth", True, "Is GPU growth allowed")

flags.DEFINE_boolean("restore_checkpoint", True, "Restore the checkpoint")

flags.DEFINE_integer("gpu_batch_size", 16, "Total batch size")
flags.DEFINE_integer("eval_gpu_batch_size", None, "Total batch size for eval")

flags.DEFINE_string("output_file", None, "Output file path")

flags.DEFINE_boolean("press_key_to_end", False, "End the program after press the key")

flags.DEFINE_string("tpu_name", None, "TPU name")

flags.DEFINE_bool("soft_device_placement", False, "If set soft device placement")

flags.DEFINE_integer("sliding_window_crop_height", None, "Sliding window crop height")
flags.DEFINE_integer("sliding_window_crop_width", None, "Sliding window crop width")

# Dataset settings

flags.DEFINE_integer("crop_height", None, "crop height")
flags.DEFINE_integer("crop_width", None, "crop width")
flags.DEFINE_integer("eval_crop_height", None, "eval crop height")
flags.DEFINE_integer("eval_crop_width", None, "eval crop width")


# Training protocol

flags.DEFINE_bool("mixed_precision", True, "Use mixed precision")
flags.DEFINE_float("sgd_momentum_rate", 0, "Momentum rate of SGD")

flags.DEFINE_integer("max_checkpoints_to_keep", 20, "How many checkpoints to keep")

flags.DEFINE_integer("train_epoch", 30, "Epoch to train")
flags.DEFINE_integer("epoch_steps", 1000, "Num steps in each epoch")
flags.DEFINE_integer("initial_epoch", 0, "Initial epoch")
flags.DEFINE_integer("shuffle", 256, "Shuffle rate")

flags.DEFINE_integer("random_seed", 0, "random seed")
flags.DEFINE_bool("training_progress_bar", True, "Show progress bar during training")

# Evaluation protocol

flags.DEFINE_multi_float("scale_rates", [1.0], "Scale rates when predicion")
flags.DEFINE_boolean("flip", False, "Use flip when prediction")
