import numpy as np

import torch
from torch.nn import functional as F
from tps.tps_sampler import TPSRandomSampler


class BatchTransform(object):
    """ Preprocessing batch of pytorch tensors
    """
    def __init__(self, image_size=[128, 128], \
            rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], \
            transsd=[0.1, 0.1], warpsd=[0.001, 0.005, 0.001, 0.01]):
        self.image_size = image_size
        self.target_sampler, self.source_sampler = \
            self._create_tps(image_size, rotsd, scalesd, transsd, warpsd)

    def exe(self, image, landmarks=None):
        #call _proc_im_pair
        batch = self._proc_im_pair(image, landmarks=landmarks)

        #call _apply_tps
        image, future_image, future_mask = self._apply_tps(batch['image'], batch['mask'])

        batch.update({'image': image, 'future_image': future_image, 'mask': future_mask})

        return batch

    #TPS
    def _create_tps(self, image_size, rotsd, scalesd, transsd, warpsd):
        """create tps sampler for target and source images"""
        target_sampler = TPSRandomSampler(
            image_size[1], image_size[0], rotsd=rotsd[0],
            scalesd=scalesd[0], transsd=transsd[0], warpsd=warpsd[:2], pad=False)
        source_sampler = TPSRandomSampler(
            image_size[1], image_size[0], rotsd=rotsd[1],
            scalesd=scalesd[1], transsd=transsd[1], warpsd=warpsd[2:], pad=False)
        return target_sampler, source_sampler

    def _apply_tps(self, image, mask):
        #expand mask to match batch size and n_dim
        mask = mask[None, None].expand(image.shape[0], -1, -1, -1)
        image = torch.cat([mask, image], dim=1)
        # shape = image.shape

        future_image = self.target_sampler.forward(image)
        image = self.source_sampler.forward(future_image)

        #reshape -- no need
        # image = image.reshape(shape)
        # future_image = future_image.reshape(shape)

        future_mask = future_image[:, 0:1, ...]
        future_image = future_image[:, 1:, ...]

        mask = image[:, 0:1, ...]
        image = image[:, 1:, ...]

        return image, future_image, future_mask

    #Process image pair
    def _proc_im_pair(self, image, landmarks=None):
        m, M = image.min(), image.max()

        height, width = self.image_size[:2]

        #crop image
        crop_percent = 0.8
        final_sz = self.image_size[0]
        resize_sz = np.round(final_sz / crop_percent).astype(np.int32)
        margin = np.round((resize_sz - final_sz) / 2.0).astype(np.int32)

        if landmarks is not None:
            original_sz = image.shape[-2:]
            landmarks = self._resize_points(
                landmarks, original_sz, [resize_sz, resize_sz])
            landmarks -= margin

        image = F.interpolate(image, \
            size=[resize_sz, resize_sz], mode='bilinear', align_corners=True)

        #take center crop
        image = image[..., margin:margin + final_sz, margin:margin + final_sz]
        image = torch.clamp(image, m, M)

        mask = self._get_smooth_mask(height, width, 10, 20) #shape HxW
        mask = mask.to(image.device)

        future_landmarks = landmarks
        # future_image = image.clone()

        batch = {}
        batch.update({'image': image, 'mask': mask, \
            'landmarks': landmarks, 'future_landmarks': future_landmarks})

        return batch

    def _resize_points(self, points, size, new_size):
        dtype = points.dtype
        device = points.device

        size = torch.tensor(size).to(device).float()
        new_size = torch.tensor(new_size).to(device).float()

        ratio = new_size / size
        points = (points.float() * ratio[None]).type(dtype)
        return points

    def _get_smooth_step(self, n, b):
        x = torch.linspace(-1, 1, n)
        y = 0.5 + 0.5 * torch.tanh(x / b)
        return y

    def _get_smooth_mask(self, h, w, margin, step):
        b = 0.4
        step_up = self._get_smooth_step(step, b)
        step_down = self._get_smooth_step(step, -b)

        def _create_strip(size):
            return torch.cat(
                [torch.zeros(margin),
                step_up,
                torch.ones(size - 2 * margin - 2 * step),
                step_down,
                torch.zeros(margin)], dim=0)

        mask_x = _create_strip(w)
        mask_y = _create_strip(h)
        mask2d = mask_y[:, None] * mask_x[None]
        return mask2d
