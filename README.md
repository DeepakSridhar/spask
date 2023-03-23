# SPASK
ECE271B project

PyTorch implementation for self-supervised part segmentation via keypoint constraints - Application to histopathology study in cancer detection.

![Alt text](misc/spask.jpg "SPASK")

## Installation

The code is developed based on Pytorch v1.8+ with TensorboardX as visualization tools. We recommend to use the same conda env as main branch to run the code:

Download Camelyon16 dataset

## Train the model

```$ CUDA_VISIBLE_DEVICES={GPU} python train.py -f exps/SPASK_LL_K2.json``` where `{GPU}` is the GPU device number.


## License

Apache 2.0 license
