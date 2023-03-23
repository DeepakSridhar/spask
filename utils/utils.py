"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import pair_confusion_matrix

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False, yx=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        if yx:
            centers.append(torch.stack((y_c, x_c), dim=0).unsqueeze(0))
        else:
            centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

def batch_get_centers(pred_softmax, yx=False):
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b], yx=yx).unsqueeze(0))
    return torch.cat(centers_list, dim=0)


def batch_get_corners(pred_softmax):
    B,C,H,W = pred_softmax.shape

    corners_list1 = []
    corners_list2 = []
    for b in range(B):
        corners_list1.append(get_corners(pred_softmax[b])[0].unsqueeze(0))
        corners_list2.append(get_corners(pred_softmax[b])[1].unsqueeze(0))
    return torch.cat(corners_list1, dim=0), torch.cat(corners_list2, dim=0)


def get_corners(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C,H,W = part_maps.shape
    corners1 = []
    corners2 = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_min, y_min, x_max, y_max = get_corner(part_map_pdf, self_ref_coord)
        corners1.append(torch.stack((x_min, y_min), dim=0).unsqueeze(0))
        corners2.append(torch.stack((x_max, y_max), dim=0).unsqueeze(0))
    return torch.cat(corners1, dim=0), torch.cat(corners2, dim=0)

def get_corner(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    x_min = (part_map * x_map).min()
    y_min = (part_map * y_map).min()
    x_max = (part_map * x_map).max()
    y_max = (part_map * y_map).max()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_min, y_min, x_max, y_max



class Colorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(n)
        print(self.cmap)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def denseCRF(img, pred):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    N,H,W = pred.shape

    d = dcrf.DenseCRF2D(W, H, N)  # width, height, nlabels
    U = unary_from_softmax(pred)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=5)

    Q = d.inference(5)
    Q = np.array(Q).reshape((N,H,W)).transpose(1,2,0)

    return Q


def argmax_onehot(x, dim=1):
    m = torch.argmax(x, dim=dim, keepdim=True)
    x = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(dim, m, 1.0)
    return x

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class, class_names=None):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class) if class_names is None else class_names, iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def adjusted_rand_score_overflow(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0
    (tn, fp), (fn, tp) = (tn / 1e4, fp / 1e4), (fn / 1e4, tp / 1e4)
    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))


def seed_worker(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def pil_loader(path, type):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(type)


def pad_if_smaller(img, size, fill=None):

    min_size = min(img.shape[:2])
    if min_size < size:
        ow, oh = img.shape[:2]
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        pad = ((padw // 2, padw - padw // 2), (padh // 2, padh - padh // 2), (0,0)) if len(img.shape) == 3 else ((padw // 2, padw - padw // 2), (padh // 2, padh - padh // 2))
        if fill is None:
            img = np.pad(img, pad, 'edge')
        else:
            img = np.pad(img, pad, 'constant', constant_values=fill)
    return img

    
