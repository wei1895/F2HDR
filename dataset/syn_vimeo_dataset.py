"""Vimeo-90K training dataset for 2-exposure HDR video reconstruction.

LDR frames are synthesized on the fly from Vimeo septuplets: a random inverse
camera-response curve maps sRGB frames to linear HDR, then alternating exposures are
applied. Three consecutive frames are used (prev, cur, nxt). Ground-truth flow is not
available for Vimeo, so ``flow_mask = 0`` disables the flow-supervision loss term.
"""
import os

import numpy as np
import torch
from imageio import imread
from torch.utils.data import Dataset

from utils.utils import hdr_to_ldr, random_crop, random_flip_lrud, read_list

np.random.seed(0)


class Syn_Vimeo_Dataset(Dataset):
    def __init__(self, root_dir, nframes=3, nexps=2, is_training=True, crop_size=256):
        assert nexps == 2, "F2HDR supports 2 exposures only"
        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps
        self.crop_size = crop_size
        self.repeat = 1

        list_name = "sep_trainlist.txt" if is_training else "sep_testlist.txt"
        self.patch_list = read_list(os.path.join(self.root_dir, list_name))
        if not is_training:
            self.patch_list = self.patch_list[:1000]

    def __getitem__(self, index):
        img_dir = os.path.join(self.root_dir, "sequences", self.patch_list[index // self.repeat])
        img_idxs = sorted(np.random.permutation(7)[:self.nframes] + 1)
        if np.random.random() > 0.5:  # inverse temporal order
            img_idxs = img_idxs[::-1]
        img_paths = [os.path.join(img_dir, "im%d.png" % idx) for idx in img_idxs]

        exposures = self._get_2exposures(index)
        n, sigma = self.sample_camera_curve()

        hdrs = []
        for img_path in img_paths:
            img = (imread(img_path).astype(np.float32) / 255.0).clip(0, 1)
            linear_img = self.apply_inv_sigmoid_curve(img, n, sigma)
            linear_img = self.discretize_to_uint16(linear_img)
            hdrs.append(linear_img)

        crop_h, crop_w = self.crop_size, self.crop_size
        hdrs = random_flip_lrud(hdrs)
        hdrs = random_crop(hdrs, [crop_h, crop_w])
        color_permute = np.random.permutation(3)
        for i in range(len(hdrs)):
            hdrs[i] = hdrs[i][:, :, color_permute]

        hdrs, ldrs = self.re_expose_ldrs(hdrs, exposures)
        ldrs_tensor, hdrs_tensor, expos_tensor = [], [], []
        for i in range(len(ldrs)):
            ldrs_tensor.append(torch.from_numpy(ldrs[i].astype(np.float32).transpose(2, 0, 1)))
            hdrs_tensor.append(torch.from_numpy(hdrs[i].astype(np.float32).transpose(2, 0, 1)))
            expos_tensor.append(torch.tensor(exposures[i]))

        flow_gts = [torch.zeros(2, crop_h, crop_w), torch.zeros(2, crop_h, crop_w)]
        flow_mask = torch.tensor(0.0)

        return {
            "hdrs": hdrs_tensor,
            "ldrs": ldrs_tensor,
            "expos": expos_tensor,
            "flow_gts": flow_gts,
            "flow_mask": flow_mask,
        }

    # ----- camera-curve synthesis ---------------------------------------------
    def sample_camera_curve(self):
        n = np.clip(np.random.normal(0.65, 0.1), 0.4, 0.9)
        sigma = np.clip(np.random.normal(0.6, 0.1), 0.4, 0.8)
        return n, sigma

    def apply_inv_sigmoid_curve(self, y, n, sigma):
        return np.power((sigma * y) / (1 + sigma - y), 1 / n)

    def discretize_to_uint16(self, img):
        max_int = 2 ** 16 - 1
        return np.uint16(img * max_int).astype(np.float32) / max_int

    def _get_2exposures(self, index):
        cur_high = np.random.uniform() > 0.5
        exposures = np.ones(self.nframes, dtype=np.float32)
        high_expo = np.random.choice([4.0, 8.0])
        start = 0 if cur_high else 1
        for i in range(start, self.nframes, 2):
            exposures[i] = high_expo
        return exposures

    def re_expose_ldrs(self, hdrs, exposures):
        mid = len(hdrs) // 2
        if exposures[mid] == 1:  # low-exposure reference
            factor = np.random.uniform(0.1, 0.8)
            anchor = hdrs[mid].max()
            new_anchor = anchor * factor
        else:                    # high-exposure reference
            percent = np.random.uniform(98, 100)
            anchor = np.percentile(hdrs[mid], percent)
            new_anchor = np.random.uniform(anchor, 1)

        new_hdrs = [(hdr / (anchor + 1e-8) * new_anchor).clip(0, 1) for hdr in hdrs]
        ldrs = [hdr_to_ldr(new_hdrs[i], exposures[i]) for i in range(len(new_hdrs))]
        return new_hdrs, ldrs

    def __len__(self):
        return len(self.patch_list) * self.repeat
