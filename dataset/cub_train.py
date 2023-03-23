"""
Code adapted from: https://github.com/akanazawa/cmr/blob/master/data/cub.py
MIT License

Copyright (c) 2018 akanazawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from . import base as base_data
from utils import transformations

# -------------- flags ------------- #
# ---------------------------------- #
if osp.exists('/scratch1/storage'):
    kData = '/scratch1/storage/CUB'
elif osp.exists('/data1/shubhtuls'):
    kData = '/data0/shubhtuls/datasets/CUB'
else:  # Savio
    kData = '/global/home/users/kanazawa/scratch/CUB'

kData = '/data/imgDB/DB/CUB200_2011/'
cub_dir = kData
cub_cache_dir = '/data8/deepak/UMR/cachedir/cub'

# -------------- Dataset ------------- #
# ------------------------------------ #
class CUBDataset(base_data.BaseDataset):
    '''
    CUB Data loader
    '''

    def __init__(self, root, split, max_iters=None, crop_size=(321, 321), center_crop=False, ignore_saliency_fg=False,
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, iou_threshold=0.3, filter_key=None):
        super(CUBDataset, self).__init__(root, filter_key=filter_key)
        #self.data_dir = opts.cub_dir
        #self.data_cache_dir = opts.cub_cache_dir
        self.root = root
        self.split = split
        self.crop_h, self.crop_w = crop_size
        self.center_crop = center_crop
        self.ignore_saliency_fg = ignore_saliency_fg
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        self.data_dir = cub_dir
        self.data_cache_dir = cub_cache_dir
        split = 'train'

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % split)

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import pdb; pdb.set_trace()
        self.filter_key = filter_key

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        annos = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.anno = []
        for anno in annos:
            img_path = anno.rel_path
            # if '031.' in img_path or '032.' in img_path or '033.' in img_path:
            self.anno.append(anno)


        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;

    # def __len__(self):
    #     return len(self.files)

    # def generate_scale_imgs(self, imgs, interp_modes):

    #     scale_imgs = ()

    #     if self.center_crop:

    #         large_crop = int(self.crop_h *1.25)
    #         margin = int((large_crop - self.crop_h)/2)

    #         for img, interp_mode in zip(imgs, interp_modes):
    #             img = cv2.resize(img, (large_crop, large_crop), interpolation = interp_mode)
    #             img = img[margin:(large_crop-margin), margin:(large_crop-margin)]
    #             scale_imgs = (*scale_imgs, img)

    #     else:
    #         f_scale_y = self.crop_h/imgs[0].shape[0]
    #         f_scale_x = self.crop_w/imgs[0].shape[1]

    #         self.scale_y, self.scale_x = f_scale_y, f_scale_x

    #         for img, interp_mode in zip(imgs, interp_modes):
    #             if img is not None:
    #                 img = cv2.resize(img, None, fx=f_scale_x, fy=f_scale_y, interpolation = interp_mode)
    #             scale_imgs = (*scale_imgs, img)

    #     return scale_imgs

    # def __getitem__(self, index):

    #     datafiles = self.files[index]
    #     image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
    #     if image is None:
    #         print(datafiles)
    #         exit()

    #     label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)


    #     if label is not None:
    #         label = label.astype(np.float32)
    #         label /= 255.0
    #         label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)

    #     size = image.shape
    #     name = datafiles["name"]

    #     # always scale to fix size

    #     image, label = self.generate_scale_imgs((image,label), (cv2.INTER_LINEAR,cv2.INTER_LINEAR))


    #     # landmarks
    #     landmarks = datafiles["landmarks"]
    #     landmarks_scale = []

    #     for kp_i in range(5):
    #         lm = landmarks[kp_i]
    #         landmarks_scale.append(torch.tensor((int(lm[0]*self.scale_x), int(lm[1]*self.scale_y))).unsqueeze(dim=0))

    #     landmarks_scale = torch.cat(landmarks_scale, dim=0)

    #     image = np.asarray(image, np.float32)
    #     image -= self.mean

    #     image = image[:, :, ::-1]  # change to BGR
    #     image = image.transpose((2, 0, 1))

    #     data_dict = {'img'     : image.copy(),
    #                  'saliency': label.copy() if label is not None else None,
    #                  'size'    : np.array(size),
    #                  'landmarks': landmarks_scale,
    #                  'name'    : name}

    #     return data_dict


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_data.base_loader(CUBDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='mask')


def sfm_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='sfm_pose')
