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
3. Install cudatoolkit 11.2, cudnn, tqdm and pillow
```
conda install cudatoolkit=11.2 -c conda-forge
conda install cudnn=8.1 -c conda-forge
conda install tqdm
conda install pillow
```
4. Install TensorFlow 2.8

```
pip install tensorflow==2.8.0
```

5. (optional) In some cases, you may have to export LD_LIBRARY_PATH to help TensorFlow find CUDA, etc.

```
export LD_LIBRARY_PATH=$CONDA_PREFIX/libs
```

You may also add this script into the conda script, see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables
