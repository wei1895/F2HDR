"""TOG13 dynamic dataset with online global alignment (qualitative test), 2-exposure.

The TOG13 sequences have no HDR ground truth; they are used for qualitative results.
Neighboring frames are globally aligned to the reference using precomputed affine
matrices (``Affine_Trans_Matrices/``). Frames are linearized with the provided camera
response function (``BaslerCRF.mat``).
"""
import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from utils.utils import apply_gamma, read_16bit_tif

np.random.seed(0)


def cvt_MToTheta(M, w, h):
    """Convert an OpenCV affine matrix to a PyTorch ``affine_grid`` theta."""
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = _get_N(w, h)
    N_inv = np.linalg.inv(N)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def _get_N(W, H):
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[1, 1] = 2.0 / H
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


class TOG13_online_align_Dataset(Dataset):
    def __init__(self, root_dir, nframes=3, nexps=2, align=True):
        assert nexps == 2, "F2HDR supports 2 exposures only"
        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps
        self.align = align

        crf_path = "/".join(root_dir.split("/")[:-1])
        self.crf = sio.loadmat(os.path.join(crf_path, "BaslerCRF.mat"))["BaslerCRF"]

        img_list, hdr_list = self._load_img_hdr_list(self.root_dir)
        e_list = self._load_exposure_list(os.path.join(self.root_dir, "Exposures.txt"), img_num=len(img_list))
        self.imgs_list, self.hdrs_list, self.expos_list = self._lists_to_paired_lists([img_list, hdr_list, e_list])
        print("[%s] totaling %d samples" % (self.__class__.__name__, len(self.imgs_list)))

    def _load_img_hdr_list(self, img_dir):
        if os.path.exists(os.path.join(img_dir, "img_hdr_list.txt")):
            img_hdr_list = np.genfromtxt(os.path.join(img_dir, "img_hdr_list.txt"), dtype="str")
            img_list, hdr_list = img_hdr_list[:, 0], img_hdr_list[:, 1]
        else:
            img_list = np.genfromtxt(os.path.join(img_dir, "img_list.txt"), dtype="str")
            hdr_list = ["None"] * len(img_list)
        img_list = [os.path.join(img_dir, p) for p in img_list]
        hdr_list = [os.path.join(img_dir, p) for p in hdr_list]
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

    def load_affine_matrices(self, img_path, h, w):
        dir_name, img_name = os.path.dirname(img_path), os.path.basename(img_path)
        cv2_match = np.genfromtxt(
            os.path.join(dir_name, "Affine_Trans_Matrices", img_name[:-4] + "_match.txt"),
            dtype=np.float32)
        n_matches = cv2_match.shape[0]
        assert n_matches == 2  # cur->prev, cur->next
        cv2_match = cv2_match.reshape(n_matches, 2, 3)
        theta = np.zeros((n_matches, 2, 3), dtype=np.float32)
        for mi in range(n_matches):
            theta[mi] = cvt_MToTheta(cv2_match[mi], w, h)
        return theta

    def __getitem__(self, index):
        ldrs, expos, matches = [], [], []
        img_paths = self.imgs_list[index]
        exposures_all = np.array(self.expos_list[index]).astype(np.float32)
        for i in range(0, self.nframes):
            img = apply_gamma(read_16bit_tif(img_paths[i], crf=self.crf), gamma=2.2)
            ldrs.append(img)
            expos.append(exposures_all[i])
            if self.align:
                matches.append(self.load_affine_matrices(img_paths[i], img.shape[0], img.shape[1]))

        ldrs_tensor, expos_tensor, matches_tensor = [], [], []
        for i in range(len(ldrs)):
            ldrs_tensor.append(torch.from_numpy(ldrs[i].astype(np.float32).transpose(2, 0, 1)))
            expos_tensor.append(torch.tensor(expos[i]))
            if self.align:
                matches_tensor.append(torch.tensor(matches[i]))

        return {"ldrs": ldrs_tensor, "expos": expos_tensor, "matches": matches_tensor}

    def __len__(self):
        return len(self.imgs_list)
