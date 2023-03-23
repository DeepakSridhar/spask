"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
modified by deepak
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loss
from utils.percep_loss import PercepLoss
from utils import batch_transform
from utils import utils
from model.feature_extraction import FeatureExtraction, featureL2Norm
from torchvision import transforms, models
from tps.rand_tps import RandTPS, IMGAUG
from visualize import Visualizer, Batch_Pred_Landmarks
from model import imm_model
from utils.unsuper_loss import ComputeLoss
from utils.vgg import VGG

import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from utils.utils import adjusted_rand_score_overflow
from torch.utils.data import DataLoader
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image

model_cam = models.efficientnet_v2_s(pretrained=True)
model_cam.eval()

targets = None#[ClassifierOutputTarget(295)]

target_layers = [model_cam.features[-1]]


IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


class PartBasisGenerator(nn.Module):
    def __init__(self, feature_dim, K, normalize=False):
        super(PartBasisGenerator, self).__init__()
        self.w = nn.Parameter(
            torch.abs(torch.cuda.FloatTensor(K, feature_dim).normal_()))
        self.normalize = normalize

    def forward(self, x=None):
        out = nn.ReLU()(self.w)
        if self.normalize:
            return featureL2Norm(out)
        else:
            return out


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


class SCOPSTrainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model

        # Initialize spatial/color transform for Equuivariance loss.
        self.tps = RandTPS(args.input_size[1], args.input_size[0],
                           batch_size=args.batch_size,
                           sigma=args.tps_sigma,
                           border_padding=args.eqv_border_padding,
                           random_mirror=args.eqv_random_mirror,
                           random_scale=(args.random_scale_low,
                                         args.random_scale_high),
                           mode=args.tps_mode).cuda(args.gpu)

        # Color Transorm.
        self.cj_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.ToTensor(), ])

        # KL divergence loss for equivariance
        self.kl = nn.KLDivLoss().cuda(args.gpu)

        # loss/ bilinear upsampling
        self.interp = nn.Upsample(
            size=(args.input_size[1], args.input_size[0]), mode='bilinear', align_corners=True)

        # Initialize feature extractor and part basis for the semantic consistency loss.
        self.zoo_feat_net = FeatureExtraction(
            feature_extraction_cnn=args.ref_net, normalization=args.ref_norm, last_layer=args.ref_layer)
        self.zoo_feat_net.eval()

        #batch data transform
        self.batch_transform = batch_transform.BatchTransform(image_size=(args.input_size[1], args.input_size[0]))
        #loss function
        self.recon_criterion = PercepLoss(args)

        self.separation_loss = loss.SeparationLoss(1, 1)

        self.part_basis_generator = PartBasisGenerator(self.zoo_feat_net.out_dim,
                                                       args.num_parts, normalize=args.ref_norm)
        self.part_basis_generator.cuda(args.gpu)
        self.part_basis_generator.train()

        if args.restore_part_basis != '':
            rr = torch.load(args.restore_part_basis)
            self.part_basis_generator.load_state_dict(rr['W_state_dict'])
                # {'w': rr['W_state_dict']})

        # Initialize optimizers.
        self.optimizer_seg = optim.SGD(self.model.optim_parameters(args),
                                       lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_seg.zero_grad()

        self.optimizer_sc = optim.SGD(self.part_basis_generator.parameters(
        ), lr=args.learning_rate_w, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_sc.zero_grad()

        # Initialize optimizers.
        if args.optimize_full_net:
            params = self.model.optim_parameters_all(args)
        else:
            params = self.model.optim_parameters_linear(args)
        self.optimizer_linear = optim.SGD(params,
                                       lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_linear.zero_grad()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_seg, milestones=[15000, 30000], gamma=0.1)

        # visualizor
        self.viz = Visualizer(args)

    def step_ll(self,batch,output,current_step,loss_type = 'vanilla'):

        bce_loss = 0

        self.optimizer_linear.zero_grad()

        gt = batch['true_lab']

        if loss_type == 'vanilla':
            bce_loss = loss.BCE_loss(output,gt)
        elif loss_type == 'cross_entropy':
            ce_loss = loss.cross_entropy_loss(output, gt)
            bce_loss = ce_loss.mean()
        else:
            bce_loss = loss.MOD_BCE_loss(output,gt)
            bce_loss = bce_loss.mean()

        bce_loss.backward()

        self.optimizer_linear.step()
        self.scheduler.step()


        if current_step % self.args.vis_interval == 0:
            print('exp = {}'.format(osp.join(self.args.snapshot_dir, self.args.exp_name)))
            print(('iter = {:8d}/{:8d}, ' + 'train bce_loss = {:.3f}').format(current_step, self.args.num_steps,bce_loss ))

        return bce_loss
    
    def step(self, batch, current_step):
        loss_con_value = 0
        loss_eqv_value = 0
        loss_lmeqv_value = 0
        loss_sc_value = 0
        loss_orthonamal_value = 0
        loss_recon_pred_value = 0
        loss_text = 0
        text_loss = 0
        loss_sep_value = 0
        loss_sal_value = 0

        self.optimizer_seg.zero_grad()
        self.optimizer_sc.zero_grad()
        adjust_learning_rate(self.optimizer_seg, current_step, self.args)

        images_cpu = batch['img']
        labels = batch['saliency'] if 'saliency' in batch.keys() else None
        edges = batch['edge'] if 'edge' in batch.keys() else None
        gts = batch['gt'] if 'gt' in batch.keys() else None

        landmarks = batch['landmarks'] if 'landmarks' in batch.keys() else None
        bbox = batch['bbox'] if 'bbox' in batch.keys() else None

        images_cj = torch.from_numpy(
            ((images_cpu.numpy() + IMG_MEAN.reshape((1, 3, 1, 1))) / 255.0).clip(0, 1.0))
        inp_img_pose = images_cj.cuda()
        deformed_batch = self.batch_transform.exe(images_cj, landmarks=landmarks)
        im, future_im, mask = deformed_batch['image'], deformed_batch['future_image'], deformed_batch['mask']
        i_shape = images_cpu.shape
        images_zoo_cpu = im.numpy()
        mean_tensor = torch.tensor(IMG_MEAN).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
        images_cpu_im = (im * 255.0) - mean_tensor
        future_images_cpu_im = (future_im * 255.0) - mean_tensor
        im = im.cuda()
        future_im = future_im.cuda()


        for b in range(images_cj.shape[0]):
            images_cj[b] = torch.from_numpy(self.cj_transform(
                images_cj[b]).numpy() * 255.0 - IMG_MEAN.reshape((1, 3, 1, 1)))
        images_cj = images_cj.cuda()

        # images = images_cpu.cuda(self.args.gpu)
        images = im
        images_tps = future_im
        feature_instance_dict, feature_part, pred_low = self.model(images, images_tps)
        pred = self.interp(pred_low)


        input_tensor_cam = images_tps.cpu().numpy()

        # # prepare for torch model_zoo models images
        zoo_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        zoo_var = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        input_tensor_cam -= zoo_mean
        input_tensor_cam /= zoo_var

        input_tensor_cam_cpu = torch.from_numpy(input_tensor_cam)
        input_tensor_cam = input_tensor_cam_cpu.cuda(self.args.gpu)

        with torch.no_grad():

            # zoo_feats = self.zoo_feat_net(images_zoo)
            zoo_feats = self.zoo_feat_net(input_tensor_cam)
            zoo_feat = torch.cat([self.interp(zoo_feat)
                                  for zoo_feat in zoo_feats], dim=1)

        with GradCAM(model=model_cam, target_layers=target_layers) as cam:
            grayscale_cams = cam(input_tensor=input_tensor_cam_cpu, targets=targets)
            back_mask = torch.from_numpy(grayscale_cams).unsqueeze(1)
        
        # saliency masking
        if not self.args.no_sal_masking and labels is not None:
            ll = back_mask.expand_as(
                    zoo_feat)
            
            zoo_feat = zoo_feat * \
                ll.cuda(self.args.gpu)

        feature_instance_tps, feature_part_tps, pred_low_tps = self.model(
            images_tps)

        pred_tps = self.interp(pred_low_tps)

        # Texture loss
        if self.args.loss_texture:
            vgg_layers = [int(i) for i in self.args.texture_layers]
            vgg_texture = VGG(layers=vgg_layers, replace_pooling = False)
            
            vgg_texture = vgg_texture.cuda()

            def gram_matrix(y):
                (b, ch, h, w) = y.size()
                features = y.view(b, ch, w * h)
                features_t = features.transpose(1, 2)
                gram = features.bmm(features_t) / (ch * h * w)
                return gram
            
            def criterion(a, b):
                return torch.mean(torch.abs((a-b)**2).view(-1))
            sh = pred_tps.shape
            mm = torch.tensor(zoo_mean).float().cuda()
            vv = torch.tensor(zoo_var).float().cuda()
            pred_kts = []
            part_tps_softmax1 = nn.Softmax(dim=1)(pred_tps)
            # normalize
            part_tps_softmax2 = part_tps_softmax1 / part_tps_softmax1.max(dim=3, keepdim=True)[
                0].max(dim=2, keepdim=True)[0]
            part_tps_softmax = ((part_tps_softmax2 > 0.1)*1).type(torch.uint8) 

            for kk in range(1, self.args.num_parts):
                texte_loss = []                
                predk = part_tps_softmax[:, kk,:,:].unsqueeze(1) * future_im
                pred_kts.append(predk)
                predk -= mm
                predk /= vv
                pred_lrud = torch.flipud(torch.fliplr(predk))
                
                vgg_part = vgg_texture.forward(predk)
                vgg_part_f = vgg_texture.forward(pred_lrud)
                gram_part = [gram_matrix(y) for y in vgg_part]
                gram_part_f = [gram_matrix(y) for y in vgg_part_f]
                for m in range(0, len(vgg_part_f)):
                    texte_loss += [criterion(gram_part[m], gram_part_f[m])]
                text_loss += sum(texte_loss)
            loss_text += self.args.lambda_sc * text_loss.data.cpu().numpy()


        loss_sc = loss.semantic_consistency_loss(
            features=zoo_feat, pred=pred_tps, basis=self.part_basis_generator())
        loss_sc_value += self.args.lambda_sc * loss_sc.data.cpu().numpy()

        # orthonomal_loss
        loss_orthonamal = loss.orthonomal_loss(self.part_basis_generator())
        loss_orthonamal_value += self. args.lambda_orthonormal * \
            loss_orthonamal.data.cpu().numpy()

        # Concentratin Loss
        loss_con = loss.concentration_loss(pred_tps)
        loss_con_value += self.args.lambda_con * loss_con.data.cpu().numpy()


        
        if isinstance(feature_instance_dict, dict):
   
            future_im_pred = feature_instance_dict['future_im_pred']
            future_im_pred_tps = feature_instance_tps['future_im_pred']
            gauss_pt = feature_instance_dict['gauss_pt']
            pose_embeddings = feature_instance_dict['pose_embeddings']

            loss_sc_recon1, losses_ = self.recon_criterion(future_im_pred, future_im)
            # optinally close the loop with two reconstructions
            # loss_sc_recon2, losses_ = self.recon_criterion(future_im_pred_tps, inp_img_pose)
            loss_sc_recon = 1 * loss_sc_recon1 
            lm_pred_unsup = (gauss_pt.detach() + 1) * i_shape[3] / 2
            loss_separation = self.separation_loss(lm_pred_unsup)
            
          
            loss_recon_pred_value += 1000 * loss_sc_recon.data.cpu().numpy()
            loss_sep_value += 0 * loss_separation.data.cpu().numpy()

            
        pred_tps = self.interp(pred_low_tps)
        pred_d = pred.detach()
        pred_d.requires_grad = False
        # no padding in the prediction space
        # pred_tps_org = self.tps(pred_d, padding_mode='zeros')
        pred_tps_org_ = self.batch_transform.exe(pred_d)
        pred_tps_org = pred_tps_org_['future_image']

        loss_eqv = self.kl(F.log_softmax(pred_tps, dim=1),
                           F.softmax(pred_tps_org, dim=1))
        loss_eqv_value += self.args.lambda_eqv * loss_eqv.data.cpu().numpy()

        centers_tps = utils.batch_get_centers(nn.Softmax(dim=1)(pred_tps)[:, 1:, :, :])
        # pred_tps_org_dif = self.tps(pred, padding_mode='zeros')
        pred_tps_org_dif_ = self.batch_transform.exe(pred_d)
        pred_tps_org_dif = pred_tps_org_dif_['future_image']
        centers_tps_org = utils.batch_get_centers(nn.Softmax(
            dim=1)(pred_tps_org_dif)[:, 1:, :, :])

        loss_lmeqv = F.mse_loss(centers_tps, centers_tps_org)
        loss_lmeqv_value += self.args.lambda_lmeqv * loss_lmeqv.data.cpu().numpy()

        # Background constraint
        loss_sal = F.mse_loss(part_tps_softmax2[:, -1, :, :], 1 - back_mask.cuda())
        loss_sal_value += 100 * loss_sal.data.cpu().numpy()

        # visualization

        if current_step % self.args.vis_interval == 0:
            with torch.no_grad():
                pred_softmax = nn.Softmax(dim=1)(pred)
                part_softmax = pred_softmax[:, 1:, :, :]
                # normalize
                part_softmax /= part_softmax.max(dim=3, keepdim=True)[
                    0].max(dim=2, keepdim=True)[0]
                self.viz.vis_images(current_step, future_images_cpu_im, images_cpu_im.cpu(
                ), labels, edges, IMG_MEAN, pred_tps.float())
                # self.viz.vis_image_only(current_step, future_im_pred_tps.cpu(), IMG_MEAN, name='Future TPS Pred Image')
                self.viz.vis_image_only(current_step, future_im_pred.cpu(), IMG_MEAN, name='Future Pred Image')
                self.viz.vis_image_only(current_step, im.cpu(), IMG_MEAN, name='GT Image')
                self.viz.vis_image_only(current_step, future_im.cpu(), IMG_MEAN, name='Target Image')
                self.viz.vis_part_heatmaps(
                    current_step, part_softmax, threshold=0.1, prefix='pred')

                if landmarks is not None:
                    self.viz.vis_landmarks(current_step, future_images_cpu_im,
                                           IMG_MEAN, pred_tps, landmarks)
                    self.viz.vis_unsup_landmarks(current_step, images_cpu_im,
                                           IMG_MEAN, lm_pred_unsup)
                if bbox is not None:
                    self.viz.vis_bboxes(current_step, bbox)

                print('saving part basis')
                torch.save({'W': self.part_basis_generator().detach().cpu(), 'W_state_dict': self.part_basis_generator.state_dict()},
                           osp.join(self.args.snapshot_dir, self.args.exp_name, 'BASIS_' + str(current_step) + '.pth'))

            self.viz.vis_losses(current_step, [self.part_basis_generator.w.mean(), self.part_basis_generator.w.std()], [
                'part_basis_mean', 'part_basis_std'])

        # sum all loss terms
        total_loss = self.args.lambda_con * loss_con \
            + self.args.lambda_eqv * loss_eqv \
            + self.args.lambda_lmeqv * loss_lmeqv \
            + self.args.lambda_sc * loss_sc \
            + self.args.lambda_orthonormal * loss_orthonamal \
            + 0 * loss_separation \
            + 100 * loss_sc_recon \
            + self.args.lambda_sc * text_loss
        
        total_loss.backward()

        # visualize loss curves
        self.viz.vis_losses(current_step,
                            [loss_con_value, loss_eqv_value, loss_lmeqv_value, loss_recon_pred_value,
                             loss_sc_value, loss_orthonamal_value, loss_text, loss_sep_value],
                            ['loss_con', 'loss_eqv', 'loss_lmeqv', 'loss_recon_pred', 'loss_sc', 'loss_orthonamal', 'loss_texture', 'loss_separation'])
        # clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradients)
        self.optimizer_seg.step()

        nn.utils.clip_grad_norm_(
            self.part_basis_generator.parameters(), self.args.clip_gradients)
        self.optimizer_sc.step()

        print('exp = {}'.format(osp.join(self.args.snapshot_dir, self.args.exp_name)))
        print(('iter = {:8d}/{:8d}, ' +
               'loss_con = {:.3f}, ' +
               'loss_eqv = {:.3f}, ' +
               'loss_lmeqv = {:.3f}, ' +
               'loss_sc = {:.3f}, ' +
               'loss_r_p = {:.3f}, ' +
               'loss_text = {:.3f}, ' +
               'loss_sep = {:.3f}, ' +
               'loss_orthonamal = {:.3f}')
              .format(current_step, self.args.num_steps,
                      loss_con_value,
                      loss_eqv_value,
                      loss_lmeqv_value,
                      loss_sc_value,
                      loss_recon_pred_value,
                      loss_text,
                      loss_sep_value,
                      loss_orthonamal_value))
        
    # additional code from https://github.com/subhc/unsup-parts
    def log_ari(self, testloader, i_iter):
        self.model.eval()
        gts = []
        preds = []
        for batch in tqdm(testloader):
            images_cpu = batch['img']
            gt = batch['seg']
            images = images_cpu.cuda(self.args.gpu)
            _, _, pred_low, _ = self.model(images)
            pred = self.interp(pred_low)

            pred_argmax = pred.argmax(dim=1)
            gts.append(gt.type(torch.int8))
            preds.append(pred_argmax.cpu().type(torch.int8))

        gts = torch.cat(gts, 0).flatten()
        preds = torch.cat(preds, 0).flatten()

        preds = preds[gts != 0]
        gts = gts[gts != 0]

        ari = adjusted_rand_score_overflow(preds, gts)
        nmi = normalized_mutual_info_score(preds, gts)
        print(f"ARI: {ari * 100: .2f}, NMI: {nmi * 100: .2f}")
        wandb.log({'data/nmi': nmi * 100}, step=i_iter)
        wandb.log({'data/ari': ari * 100}, step=i_iter)

    def log_nmi(self, trainloader, testloader, i_iter):
        self.model.eval()
        data = {}
        for k, v in self.get_pred_and_gt(trainloader).items():
            data[f"train_{k}"] = v
        for k, v in self.get_pred_and_gt(testloader).items():
            data[f"val_{k}"] = v

        nmi = normalized_mutual_info_score(data[f"val_nmi_gt"], data[f"val_nmi_pred"])
        ari = adjusted_rand_score(data[f"val_nmi_gt"], data[f"val_nmi_pred"])
        errors = np.zeros(3)
        for i in range(3):
            errors[i] = self.kp_evaluation(data, class_id=i+1)
        print(f"NMI: {nmi * 100: .2f} ARI: {ari * 100: .2f} LR1: {errors[0] * 100: .2f} LR2: {errors[1] * 100: .2f} LR3: {errors[2] * 100: .2f}")
        wandb.log({'data/nmi': nmi * 100, 'data/ari': ari * 100, 'data/regress_cls_1': errors[0] * 100, 'data/regress_cls_2': errors[1] * 100, 'data/regress_cls_3': errors[2] * 100}, step=i_iter)

    def get_pred_and_gt(self, dataset):
        loader = DataLoader(LitDataset(dataset), batch_size=8, shuffle=False, num_workers=5, drop_last=False)
        with torch.no_grad():
            all_preds = []
            all_gts = []
            all_visible = []
            all_labels = []
            all_nmi_preds = []
            all_nmi_gts = []
            for batch in tqdm(loader):
                # deal with parts and cropping
                image = batch['img'].cuda(self.args.gpu)
                mask = batch['mask'].cuda(self.args.gpu)
                parts = batch["kp"].float()
                res = self.model(image.cuda(self.args.gpu))
                part_name_mat = torch.softmax(self.interp(res[2]), dim=1)*mask
                fac = 1.0 / torch.clamp_min(part_name_mat.sum(3).sum(2), 1.0)
                center_of_mass_x = (part_name_mat * self.pos_x).sum(3).sum(2) * fac
                center_of_mass_y = (part_name_mat * self.pos_y).sum(3).sum(2) * fac
                pred_parts_raw = torch.cat([center_of_mass_x.unsqueeze(2), center_of_mass_y.unsqueeze(2)], dim=2)
                pred_parts = pred_parts_raw / part_name_mat.size(2)  # normalize by image_size
                gt_parts = (parts[:, :, :2] + 1) / 2.
                all_preds.append(pred_parts.cpu())
                all_gts.append(gt_parts.cpu())
                all_visible.append(parts[:, :, 2].cpu())
                all_labels.append(batch["label"].cpu())

                visible = parts[:, :, 2] > 0.5
                points = parts[:, :, :2].unsqueeze(2)
                part_name_mat = self.interp(res[2])
                pred_parts_loc = F.grid_sample(part_name_mat.float().cpu(), points, mode='nearest', align_corners=False)
                pred_parts_loc = torch.argmax(pred_parts_loc, dim=1).squeeze(2)
                pred_parts_loc = pred_parts_loc[visible]
                all_nmi_preds.append(pred_parts_loc.cpu().numpy())
                gt_parts_loc = torch.arange(parts.shape[1]).unsqueeze(0).repeat(parts.shape[0], 1)
                gt_parts_loc = gt_parts_loc[visible]
                all_nmi_gts.append(gt_parts_loc.cpu().numpy())

            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_gts = torch.cat(all_gts, dim=0).numpy()
            all_visible = torch.cat(all_visible, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_nmi_preds = np.concatenate(all_nmi_preds, axis=0)
            all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)

        return {"pred": all_preds, "gt": all_gts,
                "nmi_pred": all_nmi_preds, "nmi_gt": all_nmi_gts,
                "visible": all_visible, "labels": all_labels, }

    def kp_evaluation(self, data, class_id=None):
        # https://github.com/NVlabs/SCOPS/blob/master/evaluation/face_evaluation_wild.py
        test_fit_kp = np.zeros_like(data["val_gt"])

        train_pred_flat = data["train_pred"].reshape(data["train_pred"].shape[0], -1)
        val_pred_flat = data["val_pred"].reshape(data["val_pred"].shape[0], -1)

        for i in range(data["train_gt"].shape[1]):
            scaler_pred = StandardScaler()
            scaler_gt = StandardScaler()

            train_vis = data["train_visible"][:, i] > 0.5
            scaler_pred.fit(train_pred_flat[train_vis])
            scaler_gt.fit(data["train_gt"][train_vis, i, :])

            train_pred_kp_flat_transform = scaler_pred.transform(train_pred_flat[train_vis])
            train_gt_kp_flat_transform = scaler_gt.transform(data["train_gt"][train_vis, i, :])

            mdl = LinearRegression(fit_intercept=False)
            mdl.fit(train_pred_kp_flat_transform, train_gt_kp_flat_transform)

            # test
            test_vis = data["val_visible"][:, i] > 0.5
            test_pred_kp_flat_transform = scaler_pred.transform(val_pred_flat[test_vis])
            test_fit_kp[test_vis, i, :] = scaler_gt.inverse_transform(mdl.predict(test_pred_kp_flat_transform))
        mean_error_test = self.mean_error(test_fit_kp, data, "val", class_id)
        return mean_error_test

    @staticmethod
    def mean_error(fit_kp, data, mode, class_id=None):
        gt = data[f"{mode}_gt"]
        visible = data[f"{mode}_visible"]
        diff = (fit_kp - gt)
        err = np.linalg.norm(diff, axis=2)
        err *= visible
        if class_id is not None:
            class_mask = data[f"{mode}_labels"] == class_id
            err = err[class_mask]
            visible = visible[class_mask]
        return err.sum() / visible.sum()
