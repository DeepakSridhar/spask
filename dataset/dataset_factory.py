"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


def dataset_generator(args):
    
    if args.dataset.split('_')[0] == 'CelebAWild':
        from dataset.celeba_wild_dataset import CelebAWildDataset
        dataset = CelebAWildDataset
        train_dataset = dataset(args.data_dir, args.data_list, crop_size=args.input_size,
                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                center_crop=args.center_crop, ignore_saliency_fg=args.ignore_saliency_fg,
                                iou_threshold=args.celeba_th)
    elif args.dataset.split('_')[0] == 'VOC':
        from dataset.voc import VOC12SegDataset
        dataset = VOC12SegDataset
        
        train_dataset = dataset(args.data_dir, args.data_list, crop_size=args.input_size[0],
                                img_fliplr=args.random_mirror, mean=np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32),
                                )
    elif args.dataset.split('_')[0] == 'CUB':
        from dataset.cub_train import CUBDataset
        dataset = CUBDataset
        train_dataset = dataset(args.data_dir, args.data_list, crop_size=args.input_size,
                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                center_crop=args.center_crop, ignore_saliency_fg=args.ignore_saliency_fg,
                                iou_threshold=args.celeba_th)
    elif args.dataset.split('_')[0] == 'AFLW':
        from dataset.aflw_dataset import AFLWDataset
        dataset = AFLWDataset
        train_dataset = dataset(args.data_dir, args.data_list, crop_size=args.input_size,
                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                center_crop=args.center_crop, ignore_saliency_fg=args.ignore_saliency_fg,
                                split='train')
    else:
        print('Dataset [{}] does not exisit!'.format(args.dataset))
        return None
    return train_dataset
