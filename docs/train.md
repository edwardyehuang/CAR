# Train guide

## Prerequisite

1. Make sure you have installed TensorFlow 2.8
2. Make sure you have downloaded backbone weights and datasets
3. Make sure you prepared the datasets

You can found related guides in docs

## Train step by step (ResNet-50 + Self-attention + CAR)

1. Edit or make a new copy of the "docs/pascalcontext_train_resnet50_sa_car.cfg". If you choose to make a new copy, we suggest you change the "--flagfile" to an absolute path.

2. Modify "--checkpoint_dir" and "--tensorboard_dir" to the correct location.

3. Modify "--resnet50_weights_path" to the path of backbone weights you downloaded.

4. cd to "CAR" root folder. (Important for the relative path of "--flagfile").

5. Run the following script, if you are using conda, make sure you activated the correct environment:
```
python train.py --flagfile={path to your cfg file above}
```