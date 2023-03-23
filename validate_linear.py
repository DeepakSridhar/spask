"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import json
import os
import os.path as osp
import pickle
import random
import sys
import loss
from argparse import Namespace

import numpy as np

import gc

import cv2
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from sklearn.metrics import confusion_matrix

import datetime as dt

import logging

import scops_trainer
from dataset.dataset_factory import dataset_generator,val_dataset_generator
from model.model_factory import model_generator


# solve potential deadlock https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)


EXP_NAME = 'SCOPS-Test'
MODEL = 'DeepLab'
BATCH_SIZE = 10
NUM_WORKERS = 4
DATASET = 'Cam16'
VAL_DATASET = 'Cam16_val'
DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
INPUT_SIZE = '32,32'
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
NUM_CLASSES = 6
NUM_STEPS = 2000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = None
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
CLIP_GRAD_NORM = 5
VIS_TNTERVAL = 100
VAL_REPORT = 100

TPS_SIGMA = 0.01
RAND_SCALE_LOW = 0.7
RAND_SCALE_HIGH = 1.1

NUM_PARTS = 10

LAMBDA_CON = 1e-1
LAMBDA_EQV = 10.0
LAMBDA_SC = 0.1
LAMBDA_SC_GEN = 0.1

# torch.autograd.set_detect_anomaly(True)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="SCOPS: Self-supervised Co-part Segmentation")

    # load args from json files
    parser.add_argument("-f", "--arg-file", type=str, default=None,
                        help="load args from json file")

    # Exp
    parser.add_argument("--exp-name", type=str, default=MODEL,
                        help="Experiment name. Default: Part-Test")
    # Model/Data description
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DeepLab_2branch")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset selection")

    parser.add_argument("--val_dataset", type=str, default=VAL_DATASET,
                        help="Dataset selection")


    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--celeba-th", type=float, default=0.0,
                        help="iou_threshold for celebAWild")
    parser.add_argument("--trained_model", type=str, default="../imm-pytorch/checkpoint/snapshot_35.pt",
                        help="trained lm model")
    parser.add_argument("--use-mlp", action="store_true",
                        help="Whether to use MLP head during training.")

    # Data Augmentation
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")

    # Training hyper parameters
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--clip-gradients", type=float, default=CLIP_GRAD_NORM,
                        help="Clip gradients norm. Default:5.")

    # constraints weighting
    parser.add_argument("--lambda-con", type=float, default=LAMBDA_CON,
                        help="weighting parameter for concentration")
    parser.add_argument("--lambda-eqv", type=float, default=LAMBDA_EQV,
                        help="weighting parameter for equivariance")
    parser.add_argument("--lambda-lmeqv", type=float, default=LAMBDA_EQV,
                        help="weighting parameter for equivariance")

    # Equivariance setting
    parser.add_argument("--tps-mode", type=str, default='affine',
                        help="tps mode: affine/projective")
    parser.add_argument("--tps-sigma", type=float, default=TPS_SIGMA,
                        help="peturbation of tps in equivariance loss")
    parser.add_argument("--eqv-random-mirror", action="store_true",
                        help="Whether to use random mirror in equvariance.")
    parser.add_argument("--eqv-border-padding", action="store_true",
                        help="Whether to use border padding in equvariance.")
    parser.add_argument("--random-scale-low", type=float, default=RAND_SCALE_LOW,
                        help="lower bound of random scaling.")
    parser.add_argument("--random-scale-high", type=float, default=RAND_SCALE_HIGH,
                        help="higher bound of random scaling.")

    # part training config
    parser.add_argument("--ignore-saliency-fg", action="store_true",
                        help="Whether to ignore saliency foreground and only enforce BG")
    parser.add_argument("--ignore-small-parts", action="store_true",
                        help="Whether to ignore small parts.")
    parser.add_argument("--center-crop", action="store_true",
                        help="Whether to crop center (MAFL).")
    parser.add_argument("--self-ref-coord", action="store_true",
                        help="Whether to use self-referenced centroid.")
    parser.add_argument("--kp-threshold", type=int, default=6,
                        help="maximum number of missing keypoints/landmarks")

    # Texture Consistency config
    parser.add_argument('--loss_texture', type=float, default=1, help='weight of texture loss. Default=1')
    parser.add_argument('--texture_layers', nargs='+', default=['8','17','26','35'], help='vgg layers for texture. Default:[]')

    # Semantic Consistency config
    parser.add_argument("--num-parts", type=int, default=NUM_PARTS,
                        help="Number of parts")
    parser.add_argument("--lambda-sc", type=float, default=LAMBDA_SC,
                        help="weighting parameter for semantic consistency constraint")
    parser.add_argument("--lambda-sc-gen", type=float, default=LAMBDA_SC_GEN,
                    help="weighting parameter for semantic consistency of generator")
    parser.add_argument("--learning-rate-w", type=float, default=1e-3,
                        help="learning rate for DFF basis")
    parser.add_argument("--ref-net", type=str, default='vgg19',
                        help="reference feature network. default: vgg19")
    parser.add_argument("--ref-layer", type=str, default='relu5_4',
                        help="default: vgg19")
    parser.add_argument("--ref-norm", action="store_true",
                        help="normalize reference feature map")
    parser.add_argument("--lambda-orthonamal", type=float, default=1e2,
                        help="weighting parameter for DFF orthonormal loss")
    
    # Vanilla BCE
    parser.add_argument("--vanilla-bce", action="store_true",
                        help="Whether to use vanilla bce.")

    parser.add_argument("--detach-k", action="store_true",
                        help="detach k")
    parser.add_argument("--no-sal-masking", action="store_true",
                        help="disable saliency constraint")

    parser.add_argument("--resume", type=str, default='',
                        help="resume training")

    parser.add_argument("--restore-part-basis", type=str, default='',
                        help="load part basis weights")

    # Save/visualization
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--vis-interval", type=int, default=VIS_TNTERVAL,
                        help="visualization interval.")
    parser.add_argument("--val-report", type=int, default=VAL_REPORT,
                        help="val result remporting interval.")    
    parser.add_argument("--tb-dir", type=str, default='tb_logs',
                        help="tensorbard dir.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


def main():



    log = logging.getLogger() #name of logger
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    filehandler = logging.FileHandler('mylog_'+str(dt.datetime.now().strftime('%d-%m-%Y %H:%M:%S'))+'.log')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s") # set format

    handler.setFormatter(formatter) # setup format
    filehandler.setFormatter(formatter) # setup format
    log.addHandler(handler) # read to go
    log.addHandler(filehandler) # read to go

    args = get_arguments()
    res = args.resume
    args_dict = vars(args)

    if args.arg_file is not None:
        with open(args.arg_file, 'r') as f:
            arg_str = f.read()
            file_args = json.loads(arg_str)
            args_dict.update(file_args)
            args = Namespace(**args_dict)
    args.resume = res
    args_str = '{}'.format(json.dumps(vars(args), sort_keys=False, indent=4))
    print(args_str)

    if not os.path.exists(os.path.join(args.snapshot_dir, args.exp_name)):
        os.makedirs(os.path.join(args.snapshot_dir, args.exp_name))

    # save args to file
    with open(os.path.join(args.snapshot_dir, args.exp_name, 'exp_args.json'), 'w') as f:
        print(args_str, file=f)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    args.input_size = input_size

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = model_generator(args)
    model.train()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    if args.resume:
        ckpt = torch.load(args.resume)
        params = ckpt 
        model.load_state_dict(params)    

    val_dataset = val_dataset_generator(args)
    valloader = data.DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=False)

    valloader_iter = enumerate(valloader)

    print("##############################################")
    print(f"test # of samples: {len(val_dataset)}")
    print("##############################################")

    bce_loss,val_acc, out, probs = calculate_epoch_loss(valloader_iter, valloader, model, 'val')
    with open('results/test_results_'+str(args.resume.split('/')[-2])+'.pkl','wb') as f:
        pickle.dump(out, f)
    with open('results/test_probs_'+str(args.resume.split('/')[-2])+'.pkl','wb') as f:
        pickle.dump(probs, f)
    
    # np.save(out, 'results/test_results_'+str(args.exp_name)+'.npy')
    # np.save(out, 'results/test_probs_'+str(args.exp_name)+'.npy')


    print(('val bce_loss = {:.3f}, val acc {}').format(bce_loss ,val_acc))


def calculate_epoch_loss(itera, loader, model, phase = 'train',cal = 'loss'):


            temp_loss = 0
            temp_acc = 0
            count = 0
            results = []
            targets = []
            st_time = dt.datetime.now()
            if cal == 'loss':
                model.eval()
                for k in range(len(loader)):
                    try:
                        _, batch_iter = itera.__next__()
                    except:
                        itera = enumerate(loader)
                        _, batch_iter = itera.__next__()

                    with torch.no_grad():
                      result = model(batch_iter['img'].cuda())
                    results.append(result[2])
                    targets.append(batch_iter['true_lab'])
                    temp_loss += loss.BCE_loss(result,batch_iter['true_lab'])
                    
                    count += 1
            results = torch.cat(results, 0)
            targets = torch.cat(targets, 0)
            outputs = torch.argmax(results, dim=1).cpu().numpy()
            probs, _ = torch.max(results.softmax(1), dim=1)
            probs = probs.cpu().numpy()

            cm = confusion_matrix(targets, outputs)
            tn, fp, fn, tp = cm.ravel()
            pr = tp/(tp+fp)
            re = tp/(tp+fn)
            tnr = tn/(tn+fp)
            acc = (tp+tn)/(tp+fp+tn+fn)
            balacc = (re + tnr) / 2
            f1 = 2*pr*re/(pr+re)
            print("Precision: {}".format(pr))
            print("Recall: {}".format(re))
            print("F-1: {}".format(f1))
            print("Acc: {}".format(acc))
            print("Balanced Accuracy: {}".format(balacc))
            
            temp_acc = loss.accuracy(results,targets)
            gc.collect()

            print(f'time taken for {phase} : {str(dt.datetime.now() -st_time)}')

            return temp_loss/count,temp_acc, outputs, probs

if __name__ == '__main__':
    main()
