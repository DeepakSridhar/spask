"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
import pickle

def jpeg_res(filename):
    with open(filename,'rb') as img_file:
        img_file.seek(163)
        a = img_file.read(2)
        height = (a[0] << 8) + a[1]
        a = img_file.read(2)
        width = (a[0] << 8) + a[1]
    return height, width


class Cam16_val(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), center_crop=False, ignore_saliency_fg=False,
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, iou_threshold=0.3, file_name=''):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.center_crop = center_crop
        self.ignore_saliency_fg = ignore_saliency_fg
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        
        self.files = []
        # Load the dataset
        level = 4
        root =  "./data/Cam16/"
        # file_name = "balanced_dataset_shuff_level_4_size_32.pkl"
        # file_name = "testing_dataset_with_gt_level4_size128.pkl"
        # file_name = "testing_dataset_with_gt_level4_size32.pkl"
        # file_name =  "patches_level" + str(level) + "_patch_size128_unbalanced.pkl"
        with open(os.path.join(root, file_name), 'rb') as f:
            if 'testing' not in file_name:
                dataset_patches, dataset_label = pickle.load(f)
                ratio = 0.8
                self.img_ids = np.arange(int((1-ratio)*len(dataset_patches)))
                if not max_iters==None:
                    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
                num_norm = 0
                num_p = 0
                for idx in range(int(ratio*len(dataset_patches)),len(dataset_patches)):
                    if dataset_label[idx] == 0:
                        num_norm += 1
                    num_p += 1
                    
                    self.files.append({
                        "img": dataset_patches[idx],
                        "label": dataset_label[idx],
                        "name": f'{idx}',
                        "landmarks": np.zeros((5, 2)),
                    })
            else:
                dataset_patches, dataset_label, gt_dict = pickle.load(f)
                # dataset_label = np.zeros(len(dataset_patches))
                cum_dict = {}
                csum = 0
                for k,v in gt_dict.items():
                    csum += v
                    cum_dict[k] = csum
                ratio = 0.
                # print(cum_dict)
                skip1 = cum_dict[21]
                skip2 = cum_dict[90]
                self.img_ids = np.arange(int((1-ratio)*len(dataset_patches)))
                if not max_iters==None:
                    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
                num_norm = 0
                num_p = 0
                for idx in range(int(ratio*len(dataset_patches)),len(dataset_patches)):
                    if skip1 - 1 <= num_p < cum_dict[22]:
                        num_p += 1
                        continue
                    if skip2 - 1 <= num_p < cum_dict[91]:
                        num_p += 1
                        continue
                    if dataset_label[idx] == 0:
                        num_norm += 1
                    num_p += 1
                    
                    self.files.append({
                        "img": dataset_patches[idx],
                        "label": dataset_label[idx],
                        "name": f'{idx}',
                        "landmarks": np.zeros((5, 2)),
                    })
        print('Val original {} filtered {} normal patches {} tumor patches {}'.format(len(self.img_ids), len(self.files), num_norm, len(self.files)-num_norm))


    def __len__(self):
        return len(self.files)

    def generate_scale_imgs(self, imgs, interp_modes):

        scale_imgs = ()

        if self.center_crop:

            large_crop = int(self.crop_h *1.25)
            margin = int((large_crop - self.crop_h)/2)

            for img, interp_mode in zip(imgs, interp_modes):
                img = cv2.resize(img, (large_crop, large_crop), interpolation = interp_mode)
                img = img[margin:(large_crop-margin), margin:(large_crop-margin)]
                scale_imgs = (*scale_imgs, img)

        else:
            f_scale_y = self.crop_h/imgs[0].shape[0]
            f_scale_x = self.crop_w/imgs[0].shape[1]

            self.scale_y, self.scale_x = f_scale_y, f_scale_x

            for img, interp_mode in zip(imgs, interp_modes):
                if img is not None:
                    img = cv2.resize(img, None, fx=f_scale_x, fy=f_scale_y, interpolation = interp_mode)
                scale_imgs = (*scale_imgs, img)

        return scale_imgs

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = datafiles["img"]
        
        if image is None:
            print(datafiles)
            exit()

        label = datafiles["label"]
        true_lab = 0 
        if label == 0:
            label = np.zeros((128, 128))
        else:
            label = np.ones((128, 128))
            true_lab = 1

        size = image.shape
        name = datafiles["name"]

        # always scale to fix size

        image, label = self.generate_scale_imgs((image,label), (cv2.INTER_LINEAR,cv2.INTER_LINEAR))


        # landmarks
        landmarks = datafiles["landmarks"]
        landmarks_scale = []

        for kp_i in range(5):
            lm = landmarks[kp_i]
            landmarks_scale.append(torch.tensor((int(lm[0]*self.scale_x), int(lm[1]*self.scale_y))).unsqueeze(dim=0))

        landmarks_scale = torch.cat(landmarks_scale, dim=0)

        image = np.asarray(image, np.float32)
        image -= self.mean

        image = image.transpose((2, 0, 1))

        data_dict = {'img'     : image.copy(),
                     'saliency': label.copy() if label is not None else None,
                     'size'    : np.array(size),
                     'landmarks': landmarks_scale,
                     'name'    : name,
                     'true_lab': true_lab}

        return data_dict




class Cam16(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), center_crop=False, ignore_saliency_fg=False,
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, iou_threshold=0.3, file_name=''):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.center_crop = center_crop
        self.ignore_saliency_fg = ignore_saliency_fg
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        
        self.files = []
        # Load the dataset
        level = 4
        # root =  "./data/Cam16/" # "/content/drive/My Drive/Dataset"
        # file_name = "balanced_dataset_shuff_level_4_size_32.pkl"
        # file_name =  "patches_level" + str(level) + "_patch_size128_unbalanced.pkl"
        with open(os.path.join(root, file_name), 'rb') as f:
            dataset_patches, dataset_label = pickle.load(f)
        
        # # Optionally shuffle data - Shuffle two lists with same order
        # # Using zip() + * operator + shuffle()
        # temp = list(zip(dataset_patches, dataset_label))
        # random.shuffle(temp)
        # res1, res2 = zip(*temp)
        # # res1 and res2 come out as tuples, and so must be converted to lists.
        # res1, res2 = list(res1), list(res2)
        # with open(os.path.join(variable_folder, "shuffle_"+file_name), 'wb') as f:
        #     pickle.dump([res1,res2], f, protocol=-1)

        self.img_ids = np.arange(int(0.8*len(dataset_patches)))
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        for idx in range(int(0.8*len(dataset_patches))):
            self.files.append({
                "img": dataset_patches[idx],
                "label": dataset_label[idx],
                "name": f'{idx}',
                "landmarks": np.zeros((5, 2)),
            })
        print('Train original {} filtered {}'.format(len(self.img_ids), len(self.files)))

    def __len__(self):
        return len(self.files)

    def generate_scale_imgs(self, imgs, interp_modes):

        scale_imgs = ()

        if self.center_crop:

            large_crop = int(self.crop_h *1.25)
            margin = int((large_crop - self.crop_h)/2)

            for img, interp_mode in zip(imgs, interp_modes):
                img = cv2.resize(img, (large_crop, large_crop), interpolation = interp_mode)
                img = img[margin:(large_crop-margin), margin:(large_crop-margin)]
                scale_imgs = (*scale_imgs, img)

        else:
            f_scale_y = self.crop_h/imgs[0].shape[0]
            f_scale_x = self.crop_w/imgs[0].shape[1]

            self.scale_y, self.scale_x = f_scale_y, f_scale_x

            for img, interp_mode in zip(imgs, interp_modes):
                if img is not None:
                    img = cv2.resize(img, None, fx=f_scale_x, fy=f_scale_y, interpolation = interp_mode)
                scale_imgs = (*scale_imgs, img)

        return scale_imgs

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = datafiles["img"]
        
        if image is None:
            print(datafiles)
            exit()

        label = datafiles["label"]
        true_lab = 0 
        if label == 0:
            label = np.zeros((128, 128))
        else:
            label = np.ones((128, 128))
            true_lab = 1

        size = image.shape
        name = datafiles["name"]

        # always scale to fix size

        image, label = self.generate_scale_imgs((image,label), (cv2.INTER_LINEAR,cv2.INTER_LINEAR))


        # landmarks
        landmarks = datafiles["landmarks"]
        landmarks_scale = []

        for kp_i in range(5):
            lm = landmarks[kp_i]
            landmarks_scale.append(torch.tensor((int(lm[0]*self.scale_x), int(lm[1]*self.scale_y))).unsqueeze(dim=0))

        landmarks_scale = torch.cat(landmarks_scale, dim=0)

        image = np.asarray(image, np.float32)
        image -= self.mean

        image = image.transpose((2, 0, 1))

        data_dict = {'img'     : image.copy(),
                     'saliency': label.copy() if label is not None else None,
                     'size'    : np.array(size),
                     'landmarks': landmarks_scale,
                     'name'    : name,
                     'true_lab': true_lab}

        return data_dict
