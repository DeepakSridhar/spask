"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
from model.feature_extraction import featureL2Norm

softmax = nn.Softmax(dim=1)

def get_variance(part_map, x_c, y_c):

    h,w = part_map.shape
    x_map, y_map = utils.get_coordinate_tensors(h,w)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y

def concentration_loss(pred):

    pred_softmax = softmax(pred)[:,1:,:,:]
    B,C,H,W = pred_softmax.shape

    loss = 0
    epsilon = 1e-3
    centers_all = utils.batch_get_centers(pred_softmax)
    for b in range(B):
        centers = centers_all[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:] + epsilon # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map/k
            x_c, y_c = centers[c]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
    loss = loss/B
    return loss/B


def corner_loss(pred, target):

    pred_softmax = softmax(pred)[:,1:,:,:]
    B,C,H,W = pred_softmax.shape

    loss = torch.tensor(0).float().cuda()
    epsilon = 1e-3
    centers_all = utils.batch_get_centers(pred_softmax)
    for b in range(B):
        centers = centers_all[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:] + epsilon # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map/k
            x_c, y_c = centers[c]
            x_c = (x_c+1.0)/2*W
            y_c = (y_c+1.0)/2*H
            y_t, x_t = target[b, c, 0], target[b, c, 1]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            v_x = (v_x)/W
            v_y = (v_y)/H
            # print(v_x, v_y)
            x_min = torch.max(x_c - 2*torch.sqrt(v_x), torch.zeros_like(x_c))
            y_min = torch.max(y_c - 2*torch.sqrt(v_y), torch.zeros_like(y_c))
            x_max = torch.min(x_c + 2*torch.sqrt(v_x), torch.ones_like(x_c)*(W-1))
            y_max = torch.min(y_c + 2*torch.sqrt(v_y), torch.ones_like(y_c)*(H-1))
            # print(x_min, y_min, x_max, y_max, x_t, y_t)
            if x_min <= x_t <= x_max and y_min <= y_t <= y_max:
                pass
            else:
                loss = 10 + loss
    loss = loss/B
    return loss/B


def corner_loss_v1(pred, target):

    pred_softmax = softmax(pred)[:,1:,:,:]
    B,C,H,W = pred_softmax.shape

    loss = torch.tensor([0]).float().cuda()
    epsilon = 1e-3
    corners1, corners2 = utils.batch_get_corners(pred_softmax)
    for b in range(B):
        corners1_ = corners1[b]
        corners2_ = corners2[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:] + epsilon # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map/k
            x_min, y_min = corners1_[c]
            x_max, y_max = corners2_[c]
            x_min = (x_c+1.0)/2*W
            y_min = (y_c+1.0)/2*H
            x_max = (x_c+1.0)/2*W
            y_max = (y_c+1.0)/2*H
            y_t, x_t = target[b, c, 0], target[b, c, 1]
            print(x_min, y_min, x_max, y_max, x_t, y_t)
            if x_min <= x_t <= x_max and y_min <= y_t <= y_max:
                loss = 0 + loss
            else:
                loss = 10 + loss
    loss = loss/B
    return loss/B

def semantic_consistency_loss(features, pred, basis):
    # get part maps
    pred_softmax = nn.Softmax(dim=1)(pred)
    part_softmax = pred_softmax[:, 1:, :, :]

    flat_part_softmax = part_softmax.permute(
        0, 2, 3, 1).contiguous().view((-1, part_softmax.size(1)))
    flat_features = features.permute(
        0, 2, 3, 1).contiguous().view((-1, features.size(1)))

    return nn.MSELoss()(torch.mm(flat_part_softmax, basis), flat_features)


def separation_loss(pred_landmarks, threshold=0.1):
    # Compute pairwise distances between all predicted landmarks
    dists = torch.cdist(pred_landmarks, pred_landmarks, p=2)

    # Compute the separation loss
    loss = torch.sum(torch.relu(threshold - dists)**2) / (pred_landmarks.shape[0] * (pred_landmarks.shape[0]-1))

    return loss


def batched_separation_loss(pred_landmarks, threshold=2.0):
    # Compute pairwise distances between all predicted landmarks
    batch_size = pred_landmarks.shape[0]
    num_landmarks = pred_landmarks.shape[1]
    dists = torch.cdist(pred_landmarks.view(-1, num_landmarks, 2), pred_landmarks.view(-1, num_landmarks, 2), p=2)

    # Compute the separation loss
    loss = torch.sum(torch.relu(threshold - dists)**2, dim=[1, 2]) / (num_landmarks * (num_landmarks-1))

    return loss.mean()


class SeparationLoss(nn.Module):
    def __init__(self, lamda, sigma):
        super(SeparationLoss, self).__init__()
        self.lamda = lamda
        self.sigma = sigma
    
    def forward(self, landmarks):
        n = landmarks.shape[1]
        loss = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = torch.norm(landmarks[:, i, :] - landmarks[:, j, :], dim=1)
                loss += d**2 * torch.exp(-d**2 / self.sigma**2)
        return self.lamda * loss.sum()


def orthonomal_loss(w):
    K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(0, 1))

    return F.mse_loss(WWT - torch.eye(K).cuda(), torch.zeros(K, K).cuda(), size_average=False)


def BCE_loss(pred,target):

    ## BCE with logits

    criterion = torch.nn.BCEWithLogitsLoss()  

    # print("loss dims : ",pred[2].size(),target.size()) 

    y_hat = pred[2]
    one_hot = torch.nn.functional.one_hot(target,2).cuda()

    # print(y_hat.get_device())
    # print(one_hot.get_device())

    return criterion(y_hat, one_hot.float())

def MOD_BCE_loss(pred,target):
    target = target.cuda()
    ## BCE with logits

    criterion = torch.nn.BCEWithLogitsLoss()  

    # print("loss dims : ",pred[2].size(),target.size()) 

    y_hat = pred[2]

    pred_zero_one = torch.argmax(y_hat,dim= 1)

    mask1 = (pred_zero_one == 0)*1
    mask2 = (target == 1)*1
    mask = mask1*mask2

    wt = torch.exp(target-pred_zero_one) * mask + torch.ones_like(mask)

    # wt = torch.exp(target-pred_zero_one)
    one_hot = torch.nn.functional.one_hot(target,2).cuda()

    # print(y_hat.get_device())
    # print(one_hot.get_device())

    return wt*criterion(y_hat, one_hot.float())


def cross_entropy_loss(pred,target):
    target = target.cuda()
    ## BCE with logits

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.1]).cuda(), label_smoothing=0.1, reduction='none')  #weight=torch.tensor([1.0,1.1]).cuda(), label_smoothing=0.1, 

    # print("loss dims : ",pred[2].size(),target.size()) 

    y_hat = pred[2]

    pred_zero_one = torch.argmax(y_hat,dim= 1)

    wt = torch.exp(target-pred_zero_one)
    loss_unweighted = criterion(y_hat, target)

    return wt*loss_unweighted


def accuracy(y_hat,target):

    outputs = torch.argmax(y_hat, dim=1)
    return torch.sum(outputs==target.cuda())/(target.size(0))