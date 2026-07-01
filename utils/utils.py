"""General utilities: metrics, HDR I/O, tonemapping, data augmentation, LR schedule."""
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
from skimage.metrics import peak_signal_noise_ratio


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------
def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))


def read_list(list_path, ignore_head=False, sort=False):
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort()
    return lists


# ---------------------------------------------------------------------------
# HDR / LDR conversions (numpy, for datasets)
# ---------------------------------------------------------------------------
def ldr_to_hdr(img, expo, gamma=2.2):
    return (img ** gamma) / (expo + 1e-8)


def hdr_to_ldr(img, expo, gamma=2.2, stdv1=1e-3, stdv2=1e-3):
    # add a touch of noise to low/mid exposures (matches HDRFlow data synthesis)
    if expo == 1.0 or expo == 4.0:
        stdv = np.random.rand(*img.shape) * (stdv2 - stdv1) + stdv1
        noise = np.random.normal(0, stdv)
        img = (img + noise).clip(0, 1)
    img = np.power(img * expo, 1.0 / gamma)
    return img.clip(0, 1)


def tonemap(x):
    return np.log(1 + 5000 * x) / np.log(1 + 5000)


def apply_gamma(image, gamma=2.2):
    image = image.clip(1e-8, 1)
    return np.power(image, 1.0 / gamma)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def range_compressor_cuda(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)


def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i], Img[i], data_range=data_range)
    return psnr / Img.shape[0]


def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    return batch_psnr(img, imclean, data_range)


# ---------------------------------------------------------------------------
# HDR image I/O
# ---------------------------------------------------------------------------
def read_16bit_tif(img_name, crf=None):
    img = cv2.imread(img_name, -1)
    img = img[:, :, [2, 1, 0]]  # BGR -> RGB
    if crf is not None:
        img = reverse_crf(img, crf)
        img = img / crf.max()
    else:
        img = img / 65535.0
    return img


def reverse_crf(img, crf):
    img = img.astype(int)
    out = img.astype(float)
    for i in range(img.shape[2]):
        out[:, :, i] = crf[:, i][img[:, :, i]]  # crf shape [65536, 3]
    return out


def read_hdr(filename, use_cv2=True):
    if use_cv2:
        return cv2.imread(filename, -1)[:, :, ::-1].clip(0)
    raise NotImplementedError


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))
        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        # Guard fully-black pixels (brightest == 0) to avoid 0/0 -> nan.
        scaled_mantissa = np.zeros_like(brightest)
        np.divide(mantissa * 255.0, brightest, out=scaled_mantissa, where=brightest > 0)
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)


def save_hdr(path, image):
    return radiance_writer(path, image)


# ---------------------------------------------------------------------------
# Data augmentation (numpy, HWC)
# ---------------------------------------------------------------------------
def random_crop(inputs, size, margin=0):
    is_list = isinstance(inputs, list)
    if not is_list:
        inputs = [inputs]
    outputs = []
    h, w, _ = inputs[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0 + margin, h - c_h - margin)
        l = random.randint(0 + margin, w - c_w - margin)
        for img in inputs:
            outputs.append(img[t:t + c_h, l:l + c_w])
    else:
        outputs = inputs
    return outputs if is_list else outputs[0]


def random_flip_lrud(inputs):
    if np.random.random() > 0.5:
        return inputs
    is_list = isinstance(inputs, list)
    if not is_list:
        inputs = [inputs]
    outputs = []
    vertical_flip = np.random.random() > 0.5
    for img in inputs:
        flip_img = np.fliplr(img)
        if vertical_flip:
            flip_img = np.flipud(flip_img)
        outputs.append(flip_img.copy())
    return outputs if is_list else outputs[0]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def adjust_learning_rate(args, optimizer, epoch):
    splits = args.lr_decay_epochs.split(":")
    assert len(splits) == 2
    downscale_epochs = [int(e) for e in splits[0].split(",")]
    downscale_rate = float(splits[1])
    logging.info("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))
    lr = args.lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    logging.info("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_parameters(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Inference padding
# ---------------------------------------------------------------------------
class InputPadder:
    """Pad images so spatial dims are divisible by ``divis_by``."""

    def __init__(self, dims, mode="sintel", divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


# ---------------------------------------------------------------------------
# Online global alignment (TOG13 qualitative test)
# ---------------------------------------------------------------------------
def affine_warp(img, theta):
    """Warp ``img`` with an affine ``theta`` (PyTorch ``affine_grid`` convention)."""
    n, c, h, w = img.shape
    affine_grid = F.affine_grid(theta, img.shape, align_corners=False)
    invalid_mask = ((affine_grid.narrow(3, 0, 1).abs() > 1) +
                    (affine_grid.narrow(3, 1, 1).abs() > 1)) >= 1
    invalid_mask = invalid_mask.view(n, 1, h, w).float()
    img1_to_img2 = F.grid_sample(img, affine_grid, align_corners=False)
    return img * invalid_mask + img1_to_img2 * (1 - invalid_mask)


def global_align_nbr_ldrs(ldrs, matches):
    """Globally align prev/next frames to the reference for 3-frame (2-exposure) input."""
    if len(ldrs) == 3:
        match_p = matches[0][:, 1].view(-1, 2, 3)
        match_n = matches[2][:, 0].view(-1, 2, 3)
        p_to_c = affine_warp(ldrs[0], match_p)
        n_to_c = affine_warp(ldrs[2], match_n)
        return [p_to_c, ldrs[1], n_to_c]
    return 0
