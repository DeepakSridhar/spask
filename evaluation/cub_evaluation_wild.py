"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

import sys
import os.path as osp
import random


def get_dists(preds, gts):
    (batches, channels) = preds.shape[:2]
    dists = np.zeros((channels, batches), np.int32)
    for b in range(batches):
        for c in range(channels):
            if gts[b, c, 0] > 0 and gts[b, c, 1] > 0:
                dists[c,b] = ((gts[b,c] - preds[b,c]) ** 2).sum() ** 0.5
            else:
                dists[c,b] = -1
    return dists


def within_threshold(dist, outsize, thr = 0.05):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return (dist < thr * outsize).sum() / float(len(dist))
    else:
        return -1

def kp_unnormalize(H, W, kp):
    kp = kp.copy()
    kp[..., 0] = (kp[..., 0] + 1)  * (W - 1) / 2
    kp[..., 1] = (kp[..., 1] + 1)  * (H - 1) / 2
    return kp

def kp_scale(H, W, kp):
    kp = kp_unnormalize(H, W, kp)
    kp[..., 0] = kp[..., 0] / W
    kp[..., 0] = kp[..., 0] / H
    return kp

def calc_pck(preds, gts, visible=None, boxsize=128):
    
    B, nparts, _ = gts.shape
    
    preds = kp_unnormalize(boxsize, boxsize, preds)
    gts   = kp_unnormalize(boxsize, boxsize, gts)

    # for i in range(B):
    #     vis = visible[i]
    #     invis = [not i for i in vis]
    #     gts[i][invis] = 0 

    dists = get_dists(preds, gts)
    acc = np.zeros(nparts, dtype=np.float32)
    avg_ccc = 0.0
    bad_idx_count = 0

    for i in range(nparts):
        acc[i] = within_threshold(dists[i], boxsize)
        if acc[i] >= 0:
            avg_ccc = avg_ccc + acc[i]
        else:
            bad_idx_count = bad_idx_count + 1
  
    if bad_idx_count == nparts:
        return 0
    else:
        return avg_ccc / (nparts - bad_idx_count) * 100

def mean_error_norm_bbox(fit_kp, gt_kp):

    err = np.zeros(gt_kp.shape[0])

    for i in range(gt_kp.shape[0]):
        fit_keypoints = fit_kp[i,:,:].squeeze()
        gt_keypoints = gt_kp[i, :, :].squeeze()
        face_error = 0
        for k in range(gt_kp.shape[1]):
            face_error += norm(fit_keypoints[k,:]-gt_keypoints[k,:]);
        face_error = face_error/gt_kp.shape[1];

        # pupil dis
        right_pupil = gt_keypoints[0, :];
        left_pupil = gt_keypoints[1, :];

        IOD = norm(right_pupil-left_pupil);

        if IOD != 0:
            err[i] = face_error/IOD
        else:
            pass
            print('IOD = 0!')

    return err.mean()



def cub_evaluation(train_pred_kp, train_gt_kp, test_pred_kp, test_gt_kp):

    train_gt_kp_flat = train_gt_kp.reshape(train_gt_kp.shape[0], -1)
    train_pred_kp_flat = train_pred_kp.reshape(train_pred_kp.shape[0], -1)

    scaler_pred = StandardScaler()
    scaler_gt = StandardScaler()

    scaler_pred.fit(train_pred_kp_flat)
    scaler_gt.fit(train_gt_kp_flat)

    train_gt_kp_flat_transform = scaler_gt.transform(train_gt_kp_flat)
    train_pred_kp_flat_transform = scaler_pred.transform(train_pred_kp_flat)

    model = LinearRegression(fit_intercept=False)

    model.fit(train_pred_kp_flat_transform, train_gt_kp_flat_transform)

    # train err
    train_fit_kp = scaler_gt.inverse_transform(model.predict(train_pred_kp_flat_transform)).reshape(train_gt_kp.shape)
    # mean_error_train = mean_error_norm_bbox(train_fit_kp, train_gt_kp)
    mean_error_train = calc_pck(train_fit_kp, train_gt_kp)
    


    #test
    test_pred_kp_flat = test_pred_kp.reshape(test_pred_kp.shape[0], -1)
    test_pred_kp_flat_transform = scaler_pred.transform(test_pred_kp_flat)

    test_fit_kp = scaler_gt.inverse_transform(model.predict(test_pred_kp_flat_transform)).reshape(test_gt_kp.shape)
    # mean_error_test = mean_error_norm_bbox(test_fit_kp, test_gt_kp)
    # print(test_fit_kp[:3])
    mean_error_test = calc_pck(test_fit_kp, test_gt_kp)

    return mean_error_train, mean_error_test


if __name__ == "__main__":
    inds = [1,2,3,4]
    train_pred_kp = np.load(osp.join(sys.argv[1], 'train', 'pred_kp.npy'))#[:, inds, :]
    train_gt_kp = np.load(osp.join(sys.argv[1], 'train', 'gt_kp.npy'))
    # train_gt_kp += 1
    # train_gt_kp *= 64

    test_pred_kp = np.load(osp.join(sys.argv[1], 'test', 'pred_kp.npy'))#[:, inds, :]
    test_gt_kp = np.load(osp.join(sys.argv[1], 'test', 'gt_kp.npy'))
    # test_gt_kp += 1
    # test_gt_kp *= 64
    train_gt_kp = kp_scale(128, 128, train_gt_kp)#[:, random.sample([i for i in range(15)], 5), :]
    test_gt_kp = kp_scale(128, 128, test_gt_kp)#[:, random.sample([i for i in range(15)], 5), :]

    # print(test_pred_kp.shape)
    # print(test_pred_kp[:2], test_gt_kp[:2])

    mean_error_train, mean_error_test = cub_evaluation(train_pred_kp, train_gt_kp, test_pred_kp, test_gt_kp)

    
    print('train err {} test err {}'.format(mean_error_train, mean_error_test))
