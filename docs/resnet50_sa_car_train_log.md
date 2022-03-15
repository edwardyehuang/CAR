# Training log for ResNet-50 + Self-attention + CAR

Note that, we delete sensitive info from the log. We only keep the nesserary data for you to check.


```

2022-03-10 15:35:27.899 2022-03-10 15:35:27.899019: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations: AVX2 AVX512F FMA

2022-03-10 15:35:27.899 To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

2022-03-10 15:35:34.216 2022-03-10 15:35:34.216724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30985 MB memory: -> device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:1a:00.0, compute capability: 7.0

2022-03-10 15:35:34.220 2022-03-10 15:35:34.220614: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 30985 MB memory: -> device: 1, name: Tesla V100-SXM2-32GB, pci bus id: 0000:1b:00.0, compute capability: 7.0

2022-03-10 15:35:34.223 2022-03-10 15:35:34.223016: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:2 with 30985 MB memory: -> device: 2, name: Tesla V100-SXM2-32GB, pci bus id: 0000:3d:00.0, compute capability: 7.0

2022-03-10 15:35:34.225 2022-03-10 15:35:34.225358: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:3 with 30985 MB memory: -> device: 3, name: Tesla V100-SXM2-32GB, pci bus id: 0000:3e:00.0, compute capability: 7.0

2022-03-10 15:35:34.227 2022-03-10 15:35:34.227733: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:4 with 30985 MB memory: -> device: 4, name: Tesla V100-SXM2-32GB, pci bus id: 0000:88:00.0, compute capability: 7.0

2022-03-10 15:35:34.230 2022-03-10 15:35:34.230078: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:5 with 30985 MB memory: -> device: 5, name: Tesla V100-SXM2-32GB, pci bus id: 0000:89:00.0, compute capability: 7.0

2022-03-10 15:35:34.232 2022-03-10 15:35:34.232428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:6 with 30985 MB memory: -> device: 6, name: Tesla V100-SXM2-32GB, pci bus id: 0000:b1:00.0, compute capability: 7.0

2022-03-10 15:35:34.234 2022-03-10 15:35:34.234794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:7 with 30985 MB memory: -> device: 7, name: Tesla V100-SXM2-32GB, pci bus id: 0000:b2:00.0, compute capability: 7.0

2022-03-10 15:35:34.290 INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1', '/job:localhost/replica:0/task:0/device:GPU:2', '/job:localhost/replica:0/task:0/device:GPU:3', '/job:localhost/replica:0/task:0/device:GPU:4', '/job:localhost/replica:0/task:0/device:GPU:5', '/job:localhost/replica:0/task:0/device:GPU:6', '/job:localhost/replica:0/task:0/device:GPU:7')

2022-03-10 15:35:34.290 I0310 15:35:34.290730 140242967754560 mirrored_strategy.py:374] Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1', '/job:localhost/replica:0/task:0/device:GPU:2', '/job:localhost/replica:0/task:0/device:GPU:3', '/job:localhost/replica:0/task:0/device:GPU:4', '/job:localhost/replica:0/task:0/device:GPU:5', '/job:localhost/replica:0/task:0/device:GPU:6', '/job:localhost/replica:0/task:0/device:GPU:7')

2022-03-10 15:35:34.306 INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK

2022-03-10 15:35:34.306 Your GPUs will likely run quickly with dtype policy mixed_float16 as they all have compute capability of at least 7.0

2022-03-10 15:35:34.306 I0310 15:35:34.306518 140242967754560 device_compatibility_check.py:123] Mixed precision compatibility check (mixed_float16): OK

2022-03-10 15:35:34.306 Your GPUs will likely run quickly with dtype policy mixed_float16 as they all have compute capability of at least 7.0

2022-03-10 15:35:37.178 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.178 I0310 15:35:37.178275 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.182 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.182 I0310 15:35:37.182794 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.188 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.188 I0310 15:35:37.188175 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.189 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.189 I0310 15:35:37.189451 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.191 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.191 I0310 15:35:37.191755 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.199 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.199 I0310 15:35:37.199181 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.234 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.234 I0310 15:35:37.234322 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.235 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.235 I0310 15:35:37.235548 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.240 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.240 I0310 15:35:37.240313 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.241 INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:37.241 I0310 15:35:37.241498 140242967754560 cross_device_ops.py:616] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).

2022-03-10 15:35:42.649 Use the random seed "0"

2022-03-10 15:35:42.649 Processed augments = ['RandomScaleAugment', 'RandomBrightnessAugment', 'PadAugment', 'RandomCropAugment', 'RandomFlipAugment']

2022-03-10 15:35:42.649 Processed augments = ['PadAugment']

2022-03-10 15:35:42.649 Processed augments = ['RandomScaleAugment', 'RandomBrightnessAugment', 'PadAugment', 'RandomCropAugment', 'RandomFlipAugment']

2022-03-10 15:35:42.649 Processed augments = ['PadAugment']

2022-03-10 15:35:42.649 ------General settings------

2022-03-10 15:35:42.649 ------head_name = nl

2022-03-10 15:35:42.649 ------apply_car = True

2022-03-10 15:35:42.649 ------apply_car_convs = True

2022-03-10 15:35:42.649 ------use_multi_lr = False

2022-03-10 15:35:42.649 ------use_aux_loss = False

2022-03-10 15:35:42.650 ------aux_loss_rate = 0.2

2022-03-10 15:35:42.650

2022-03-10 15:35:42.650 ------Baseline settings------

2022-03-10 15:35:42.650 ------train_mode = True

2022-03-10 15:35:42.650 ------baseline_mode = False

2022-03-10 15:35:42.650 ------replace_2nd_last_conv = True

2022-03-10 15:35:42.650

2022-03-10 15:35:42.650 ------CAR settings------

2022-03-10 15:35:42.650 ------train_mode = True

2022-03-10 15:35:42.650 ------use_intra_class_loss = True

2022-03-10 15:35:42.650 ------use_inter_class_loss = True

2022-03-10 15:35:42.650 ------intra_class_loss_rate = 1.0

2022-03-10 15:35:42.650 ------inter_class_loss_rate = 1.0

2022-03-10 15:35:42.650 ------use_batch_class_center = True

2022-03-10 15:35:42.650 ------use_last_class_center = False

2022-03-10 15:35:42.650 ------last_class_center_decay = 0.9

2022-03-10 15:35:42.650 ------pooling_rates = [1]

2022-03-10 15:35:42.650 ------inter_c2c_loss_threshold = 0.5

2022-03-10 15:35:42.650 ------inter_c2p_loss_threshold = 0.25

2022-03-10 15:35:42.650 ------intra_class_loss_remove_max = False

2022-03-10 15:35:42.650 ------use_inter_c2c_loss = True

2022-03-10 15:35:42.650 ------use_inter_c2p_loss = True

2022-03-10 15:35:42.650 ------filters = 512

2022-03-10 15:35:42.650 ------apply_convs = True

2022-03-10 15:35:42.650 ------num_class = 59

2022-03-10 15:35:42.650 ------ignore_label = 0

2022-03-10 15:35:42.650 Load backbone weights /weights/resnet50_bn.h5 as H5 format

2022-03-10 15:35:42.650 Epoch 1/30

2022-03-10 15:35:43.351 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.351 I0310 15:35:43.351560 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.433 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.433 I0310 15:35:43.433415 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.511 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.511 I0310 15:35:43.511723 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.924 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:43.924 I0310 15:35:43.924124 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.004 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.004 I0310 15:35:44.003971 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.081 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.081 I0310 15:35:44.080997 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.492 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.492 I0310 15:35:44.492279 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.571 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.571 I0310 15:35:44.571107 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.649 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:44.649 I0310 15:35:44.649311 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:45.089 INFO:tensorflow:batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:35:45.089 I0310 15:35:45.089460 140242967754560 cross_device_ops.py:897] batch_all_reduce: 1 all-reduces with algorithm = nccl, num_packs = 1

2022-03-10 15:37:09.076 WARNING:tensorflow:AutoGraph could not transform <bound method SegMetricWrapper.update_state of <iseg.metrics.seg_metric_wrapper.SegMetricWrapper object at 0x7f8b38667a30>> and will run it as-is.

2022-03-10 15:46:23.944 1000/1000 - 641s - loss: 1.6655 - IOU: 0.3119 - g_1_orl: 0.6554 - g_1_sal: 0.0152 - val_loss: 1.4782 - val_IOU: 0.2176 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 641s/epoch - 641ms/step

2022-03-10 15:46:23.961 Epoch 2/30

2022-03-10 15:50:42.243 1000/1000 - 258s - loss: 1.2855 - IOU: 0.4477 - g_1_orl: 0.5738 - g_1_sal: 0.0128 - val_loss: 1.2229 - val_IOU: 0.2529 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 15:50:42.259 Epoch 3/30

2022-03-10 15:55:00.815 1000/1000 - 259s - loss: 1.1142 - IOU: 0.5215 - g_1_orl: 0.5141 - g_1_sal: 0.0143 - val_loss: 1.0256 - val_IOU: 0.3231 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 15:55:00.833 Epoch 4/30

2022-03-10 15:59:19.609 1000/1000 - 259s - loss: 0.9693 - IOU: 0.5906 - g_1_orl: 0.4640 - g_1_sal: 0.0150 - val_loss: 0.8007 - val_IOU: 0.3954 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 15:59:19.626 Epoch 5/30

2022-03-10 16:03:38.409 1000/1000 - 259s - loss: 0.8921 - IOU: 0.6356 - g_1_orl: 0.4408 - g_1_sal: 0.0151 - val_loss: 0.7731 - val_IOU: 0.4109 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 16:03:38.426 Epoch 6/30

2022-03-10 16:07:57.237 1000/1000 - 259s - loss: 0.8180 - IOU: 0.6836 - g_1_orl: 0.4125 - g_1_sal: 0.0155 - val_loss: 0.7768 - val_IOU: 0.4170 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 16:07:57.253 Epoch 7/30

2022-03-10 16:12:14.573 1000/1000 - 257s - loss: 0.7678 - IOU: 0.7151 - g_1_orl: 0.3965 - g_1_sal: 0.0153 - val_loss: 0.7795 - val_IOU: 0.4414 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 16:12:14.589 Epoch 8/30

2022-03-10 16:16:33.894 1000/1000 - 259s - loss: 0.7199 - IOU: 0.7408 - g_1_orl: 0.3766 - g_1_sal: 0.0152 - val_loss: 0.7376 - val_IOU: 0.4585 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 16:16:33.910 Epoch 9/30

2022-03-10 16:20:53.113 1000/1000 - 259s - loss: 0.6838 - IOU: 0.7652 - g_1_orl: 0.3635 - g_1_sal: 0.0151 - val_loss: 0.7422 - val_IOU: 0.4633 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 16:20:53.133 Epoch 10/30

2022-03-10 16:25:11.552 1000/1000 - 258s - loss: 0.6604 - IOU: 0.7784 - g_1_orl: 0.3557 - g_1_sal: 0.0147 - val_loss: 0.7090 - val_IOU: 0.4745 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 16:25:11.568 Epoch 11/30

2022-03-10 16:29:27.745 1000/1000 - 256s - loss: 0.6398 - IOU: 0.7914 - g_1_orl: 0.3475 - g_1_sal: 0.0148 - val_loss: 0.7119 - val_IOU: 0.4695 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 256s/epoch - 256ms/step

2022-03-10 16:29:27.762 Epoch 12/30


2022-03-10 16:33:44.134 1000/1000 - 256s - loss: 0.6287 - IOU: 0.8006 - g_1_orl: 0.3432 - g_1_sal: 0.0147 - val_loss: 0.7334 - val_IOU: 0.4729 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 256s/epoch - 256ms/step

2022-03-10 16:33:44.150 Epoch 13/30

2022-03-10 16:37:59.711 1000/1000 - 256s - loss: 0.6022 - IOU: 0.8135 - g_1_orl: 0.3320 - g_1_sal: 0.0144 - val_loss: 0.7041 - val_IOU: 0.4770 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 256s/epoch - 256ms/step

2022-03-10 16:37:59.728 Epoch 14/30

2022-03-10 16:42:17.378 1000/1000 - 258s - loss: 0.5824 - IOU: 0.8238 - g_1_orl: 0.3253 - g_1_sal: 0.0142 - val_loss: 0.6997 - val_IOU: 0.4856 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 16:42:17.394 Epoch 15/30

2022-03-10 16:46:32.948 1000/1000 - 256s - loss: 0.5701 - IOU: 0.8334 - g_1_orl: 0.3204 - g_1_sal: 0.0140 - val_loss: 0.7294 - val_IOU: 0.4785 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 256s/epoch - 256ms/step

2022-03-10 16:46:32.963 Epoch 16/30

2022-03-10 16:50:48.827 1000/1000 - 256s - loss: 0.5551 - IOU: 0.8370 - g_1_orl: 0.3131 - g_1_sal: 0.0139 - val_loss: 0.6959 - val_IOU: 0.4900 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 256s/epoch - 256ms/step

2022-03-10 16:50:48.844 Epoch 17/30

2022-03-10 16:55:05.927 1000/1000 - 257s - loss: 0.5417 - IOU: 0.8457 - g_1_orl: 0.3076 - g_1_sal: 0.0137 - val_loss: 0.6978 - val_IOU: 0.4873 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 16:55:05.944 Epoch 18/30

2022-03-10 16:59:23.388 1000/1000 - 257s - loss: 0.5292 - IOU: 0.8517 - g_1_orl: 0.3000 - g_1_sal: 0.0136 - val_loss: 0.6908 - val_IOU: 0.4931 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 16:59:23.404 Epoch 19/30

2022-03-10 17:03:41.100 1000/1000 - 258s - loss: 0.5197 - IOU: 0.8562 - g_1_orl: 0.2973 - g_1_sal: 0.0134 - val_loss: 0.6791 - val_IOU: 0.4959 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 17:03:41.117 Epoch 20/30

2022-03-10 17:07:58.579 1000/1000 - 257s - loss: 0.5132 - IOU: 0.8593 - g_1_orl: 0.2956 - g_1_sal: 0.0132 - val_loss: 0.6812 - val_IOU: 0.4971 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 17:07:58.597 Epoch 21/30

2022-03-10 17:12:16.246 1000/1000 - 258s - loss: 0.5051 - IOU: 0.8631 - g_1_orl: 0.2921 - g_1_sal: 0.0131 - val_loss: 0.6893 - val_IOU: 0.4936 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 17:12:16.262 Epoch 22/30

2022-03-10 17:16:32.792 1000/1000 - 257s - loss: 0.4976 - IOU: 0.8643 - g_1_orl: 0.2879 - g_1_sal: 0.0130 - val_loss: 0.6915 - val_IOU: 0.4992 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 17:16:32.809 Epoch 23/30

2022-03-10 17:20:51.113 1000/1000 - 258s - loss: 0.4888 - IOU: 0.8706 - g_1_orl: 0.2828 - g_1_sal: 0.0129 - val_loss: 0.6844 - val_IOU: 0.5007 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 17:20:51.130 Epoch 24/30

2022-03-10 17:25:09.645 1000/1000 - 259s - loss: 0.4865 - IOU: 0.8705 - g_1_orl: 0.2831 - g_1_sal: 0.0128 - val_loss: 0.6941 - val_IOU: 0.4995 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 259s/epoch - 259ms/step

2022-03-10 17:25:09.662 Epoch 25/30

2022-03-10 17:29:26.894 1000/1000 - 257s - loss: 0.4783 - IOU: 0.8735 - g_1_orl: 0.2771 - g_1_sal: 0.0127 - val_loss: 0.6941 - val_IOU: 0.5011 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 17:29:26.911 Epoch 26/30

2022-03-10 17:33:44.527 1000/1000 - 258s - loss: 0.4761 - IOU: 0.8744 - g_1_orl: 0.2773 - g_1_sal: 0.0126 - val_loss: 0.6908 - val_IOU: 0.5031 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 258s/epoch - 258ms/step

2022-03-10 17:33:44.544 Epoch 27/30

2022-03-10 17:38:01.709 1000/1000 - 257s - loss: 0.4733 - IOU: 0.8770 - g_1_orl: 0.2761 - g_1_sal: 0.0126 - val_loss: 0.6834 - val_IOU: 0.5039 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 17:38:01.725 Epoch 28/30

2022-03-10 17:42:18.338 1000/1000 - 257s - loss: 0.4702 - IOU: 0.8803 - g_1_orl: 0.2753 - g_1_sal: 0.0125 - val_loss: 0.6850 - val_IOU: 0.5039 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 17:42:18.354 Epoch 29/30

2022-03-10 17:46:34.207 1000/1000 - 256s - loss: 0.4685 - IOU: 0.8797 - g_1_orl: 0.2736 - g_1_sal: 0.0126 - val_loss: 0.6866 - val_IOU: 0.5048 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 256s/epoch - 256ms/step

2022-03-10 17:46:34.224 Epoch 30/30

2022-03-10 17:50:51.283 1000/1000 - 257s - loss: 0.4658 - IOU: 0.8788 - g_1_orl: 0.2724 - g_1_sal: 0.0124 - val_loss: 0.6831 - val_IOU: 0.5050 - val_g_1_orl: 0.0000e+00 - val_g_1_sal: 0.0000e+00 - 257s/epoch - 257ms/step

2022-03-10 17:50:52.285 /usr/local/lib/python3.8/site-packages/keras/engine/training.py:2034: UserWarning: Metric SegMetricWrapper implements a `reset_states()` method; rename it to `reset_state()` (without the final "s"). The name `reset_states()` has been deprecated to improve API consistency.

```