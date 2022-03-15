# Install TensorFlow 2.8  + Pillow + Tqdm guide

For training, we recommend using the NVIDIA GPU with architecture ≥ volta (e.g. v100, any 20 or 30 series).

For NVIDIA GPU drvier, we recommend using the lastest version.

## Installation step by step

### For conda user (recommend)

1. Create new conda environment
```
conda create -n tf28 python=3.8
```
2. Activated the new environment
```
conda activate tf28
```
3. Install cudatoolkit 11.3, cudnn, tqdm and pillow
```
conda install cudatoolkit=11.3
conda install cudnn
conda install tqdm
conda install pillow
```
4. Install TensorFlow 2.8

```
pip install tensorflow==2.8.0
```

