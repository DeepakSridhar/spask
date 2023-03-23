# SPASK
ECE271B project

PyTorch implementation for self-supervised part segmentation via keypoint constraints - Application to histopathology study in cancer detection.

![Alt text](misc/spask.jpg "SPASK")

## Installation

The code is developed based on Pytorch v1.8+ with TensorboardX as visualization tools. We recommend to use conda env to run the code:

```
$ conda env create python3 -n spask_env
$ source spask_env/bin/activate
(spask_env)$ pip install -r requirements.txt
```

To deactivate the virtual environment, run `$conda deactivate`. To activate the environment again, run `$ conda activate spask_env`.

```$ ./download_CelebA.sh```

Download CelebA unaligned from [here](https://drive.google.com/open?id=0B7EVK8r0v71peklHb0pGdDl6R28).

## Train the model

```$ CUDA_VISIBLE_DEVICES={GPU} python train.py -f exps/SPASK_K8_train_geometric.json``` where `{GPU}` is the GPU device number.

## CAMELYON16 Dataset

The code for experiments on Camelyon16 dataset is in [cam16](https://github.com/DeepakSridhar/spask/tree/cam16) folder

## References
Code is largely based on SCOPS
[paper](https://varunjampani.github.io/papers/hung19_spask.pdf)

[supplementary](https://varunjampani.github.io/papers/hung19_spask_supp.pdf)

## License

Apache 2.0 license
