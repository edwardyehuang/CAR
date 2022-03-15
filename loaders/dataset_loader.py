import tensorflow as tf

import iseg.static_strings as ss

from absl import flags

from common_flags import FLAGS

from ids.dataset import Dataset

from ids.pascal_context import PascalContext
from ids.cityscapes_fine import CityScapesFine
from ids.cocostuff import Cocostuff
from ids.cocostuff10k import Cocostuff10K


import iseg.data_process.utils as dataprocess
from iseg.utils import *

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("train_datasets", [ss.PASCALCONTEXT], "name of the train datasets")
flags.DEFINE_multi_string("val_datasets", [], "name of the val datasets")
flags.DEFINE_string("predict_dataset", ss.PASCALCONTEXT, "name of the predict datasets")

flags.DEFINE_string("pascalcontext_path", None, "Pascal context path")
flags.DEFINE_string("cityscapesfine_path", None, "Cityscapes fine path")
flags.DEFINE_string("cocostuff_path", None, "coco stuff path")
flags.DEFINE_string("cocostuff10k_path", None, "coco stuff 1k path")

flags.DEFINE_boolean("trainval", False, "use train val set together")
flags.DEFINE_boolean("use_tfrecord", True, "Use tfrecord")

flags.DEFINE_boolean("random_brightness", True, "Use random brightness")
flags.DEFINE_boolean("photo_metric_distortion", False, "Use photo metric distortion")


def load_dataset_from_flags():

    train_ds, _, num_class, ignore_label, class_weights, train_size, _ = __process_dataests(FLAGS.train_datasets)
    _, val_ds, _, _, _, val_size, val_image_count = __process_dataests(FLAGS.val_datasets)

    return train_ds, val_ds, num_class, ignore_label, class_weights, train_size, val_size, val_image_count


def __normalize_image_value_range_fn(image, label):

    return dataprocess.normalize_value_range(image, FLAGS.backbone), label


def __process_dataests(datasets_names):

    if len(datasets_names) == 0:
        return None, None, None, None, None, None, None

    datasets = __dataset_names_to_datasets(datasets_names)
    datasets = __replcae_dataset_settings(datasets)

    datasets_count = len(datasets)

    if datasets_count <= 0:
        raise ValueError("Incorrect dataset names")

    num_class = datasets[0].num_class
    ignore_label = datasets[0].ignore_label
    image_size = (datasets[0].crop_height, datasets[0].crop_width)
    class_weights = datasets[0].class_weights

    _train_ds, _val_ds = datasets[0].load_tf_data()
    _val_image_count = datasets[0].val_image_count

    for i in range(1, datasets_count):
        dataset: Dataset = datasets[i]

        if dataset.num_class != num_class:
            raise ValueError("Number of classes are not same.")

        if dataset.ignore_label != ignore_label:
            raise ValueError("Ignore labels are not same.")

        train_ds, val_ds = dataset.load_tf_data()

        val_image_count = dataset.val_image_count

        if train_ds is not None:
            _train_ds = train_ds if _train_ds is None else _train_ds.concatenate(train_ds)

        if val_ds is not None:
            _val_ds = val_ds if _val_ds is None else _val_ds.concatenate(val_ds)
            _val_image_count += val_image_count

    _train_ds = _train_ds.map(__normalize_image_value_range_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if _val_ds is not None:
        _val_ds = _val_ds.map(__normalize_image_value_range_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return _train_ds, _val_ds, num_class, ignore_label, class_weights, image_size, _val_image_count


def __dataset_names_to_datasets(datasets_names):
    datasets = []

    for dataset_name in datasets_names:
        datasets.append(dataset_name_to_dataset(dataset_name))

    return datasets


def dataset_name_to_dataset(dataset_name):

    dataset = None

    if dataset_name == ss.PASCALCONTEXT:
        dataset = PascalContext(FLAGS.pascalcontext_path)
    elif dataset_name == ss.CITYSCAPESFINE:
        dataset = CityScapesFine(FLAGS.cityscapesfine_path)
    elif dataset_name == ss.COCOSTUFF:
        dataset = Cocostuff(FLAGS.cocostuff_path)
    elif dataset_name == ss.COCOSTUFF10K:
        dataset = Cocostuff10K(FLAGS.cocostuff10k_path)
    else:
        raise ValueError("Not supported dataset = {}".format(dataset_name))

    return dataset


def __replcae_dataset_settings(datasets):

    for dataset in datasets:

        dataset: Dataset = dataset

        dataset.mean_pixel = dataprocess.get_mean_pixel(FLAGS.backbone)  # Important for data processing

        dataset.crop_height = default_if_not(FLAGS.crop_height, dataset.crop_height)
        dataset.crop_width = default_if_not(FLAGS.crop_width, dataset.crop_width)
        dataset.eval_crop_height = default_if_not(FLAGS.eval_crop_height, dataset.eval_crop_height)
        dataset.eval_crop_width = default_if_not(FLAGS.eval_crop_width, dataset.eval_crop_width)

        dataset.trainval = FLAGS.trainval
        dataset.use_tfrecord = FLAGS.use_tfrecord
        dataset.random_brightness = FLAGS.random_brightness

        dataset.photo_metric_distortion = FLAGS.photo_metric_distortion

    return datasets