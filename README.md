# MonoSGC

## Introduction

This is the PyTorch implementation of the paper MonoSGC.

## Overview

- [Installation](#installation)
- [Getting Started](#getting-started)

## Installation

### Installation Steps

a. Clone this repository.

```shell
git clone https://github.com/JiYinshuai/MonoSGC
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
  
  ```shell
  pip install torch==1.10.0 torchvision==0.11.1 pyyaml scikit-image opencv-python numba tqdm
  ```

* We test this repository on Nvidia 3090 GPUs and Ubuntu 18.04. You can also follow the install instructions in [GUPNet](https://github.com/SuperMHP/GUPNet) (This respository is based on it) to perform experiments with lower PyTorch/GPU versions.

## Getting Started

### Dataset Preparation

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
MonoSGC
├── data
│   │── KITTI3D
|   │   │── training
|   │   │   ├──calib & label_2 & image_2 & depth_dense
|   │   │── testing
|   │   │   ├──calib & image_2
├── config
├── ...
```

* You can also choose to link your KITTI dataset path by
  
  ```
  KITTI_DATA_PATH=~/data/kitti_object
  ln -s $KITTI_DATA_PATH ./data/KITTI3D
  ```

* To ease the usage, we provide the pre-generated dense depth files at: [Google Drive](https://drive.google.com/file/d/1mlHtG8ZXLfjm0lSpUOXHulGF9fsthRtM/view?usp=sharing) 

### Training & Testing

#### Test and evaluate the pretrained models

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config config/kitti.yaml -e   
```

#### Train a model

```shell
CUDA_VISIBLE_DEVICES=0,1 python tools/train_val.py --config config/kitti.yaml
```
