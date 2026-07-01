"""Synthetic test dataset (Cinematic Video / HDR_Synthetic_Test_Dataset), 2-exposure.

Provides 3 consecutive 16-bit TIFF LDR frames with their exposures and the HDR ground
truth aligned to the reference (middle) frame.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import read_16bit_tif, read_hdr

np.random.seed(0)


class Syn_Test_Dataset(Dataset):
    def __init__(self, root_dir, nframes=3, nexps=2):
        assert nexps == 2, "F2HDR supports 2 exposures only"
        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps
        self.scene_list = np.genfromtxt(os.path.join(self.root_dir, "scenes_2expo.txt"), dtype="str")
        if self.scene_list.ndim == 0:
            self.scene_list = self.scene_list[None]

        self.expos_list, self.img_list, self.hdrs_list = [], [], []
        for scene in self.scene_list:
            img_dir = os.path.join(self.root_dir, "Images", scene)
            img_list, hdr_list = self._load_img_hdr_list(img_dir)
            e_list = self._load_exposure_list(os.path.join(img_dir, "Exposures.txt"), img_num=len(img_list))
            img_list, hdr_list, e_list = self._lists_to_paired_lists([img_list, hdr_list, e_list])
            self.expos_list += e_list
            self.img_list += img_list
            self.hdrs_list += hdr_list

        print("[%s] totaling %d samples" % (self.__class__.__name__, len(self.img_list)))

    def _load_img_hdr_list(self, img_dir):
        scene_list = np.genfromtxt(os.path.join(img_dir, "img_list.txt"), dtype="str")
        img_list = [os.path.join(img_dir, "%s.tif" % name) for name in scene_list]
        hdr_list = [os.path.join(img_dir, "%s.hdr" % name) for name in scene_list]
        return img_list, hdr_list

    def _load_exposure_list(self, expos_path, img_num):
        expos = np.genfromtxt(expos_path, dtype="float")
        expos = np.power(2, expos - expos.min()).astype(np.float32)
        return np.tile(expos, int(img_num / len(expos) + 1))[:img_num]

    def _lists_to_paired_lists(self, lists):
        paired_lists = []
        for l in lists:
            if self.nexps == 2 and self.nframes == 3:
                l = l[1:-1]
            paired_list = [l[: len(l) - self.nframes + 1]]
            for j in range(1, self.nframes):
                paired_list.append(l[j: len(l) - self.nframes + 1 + j])
            paired_lists.append(np.stack(paired_list, 1).tolist())
        return paired_lists

    def __getitem__(self, index):
        ldrs, expos = [], []
        img_paths, hdr_path = self.img_list[index], self.hdrs_list[index][1]
        exposures_all = np.array(self.expos_list[index]).astype(np.float32)
        for i in range(0, 3):
            ldrs.append(read_16bit_tif(img_paths[i]))
            expos.append(exposures_all[i])

        hdr = read_hdr(hdr_path)
        hdr_tensor = torch.from_numpy(hdr.astype(np.float32).transpose(2, 0, 1))

        ldrs_tensor, expos_tensor = [], []
        for i in range(len(ldrs)):
            ldrs_tensor.append(torch.from_numpy(ldrs[i].astype(np.float32).transpose(2, 0, 1)))
            expos_tensor.append(torch.tensor(expos[i]))

        return {
            "hdr_path": hdr_path.split("/")[-1],
            "hdr": hdr_tensor,
            "ldrs": ldrs_tensor,
            "expos": expos_tensor,
        }

    def __len__(self):
        return len(self.img_list)
