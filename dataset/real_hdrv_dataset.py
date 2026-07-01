"""Real-HDRV test dataset (2-exposure), with configurable reference-exposure mode.

The Real-HDRV test split (``Test_Compack_8frames_50scenes``) is organized as::

    <root>/<scene>/Input_0/frame_xx.tif   # SHORT exposure stream  (low)
    <root>/<scene>/Input_1/frame_xx.tif   # LONG  exposure stream  (high)
    <root>/<scene>/GT/GT_frame_xx.hdr     # HDR ground truth per frame

Each scene has 8 frames. Unlike the DeepHDRVideo benchmark, Real-HDRV ships **no
exposure ``.txt``** -- but every frame is available at *both* exposures, so the
alternating-exposure capture can be simulated by picking, per frame, either the short
or the long stream.

Exposure values are not provided either. We use the short frame as the unit exposure
(1.0) and the long frame as 8.0. This 1:8 (3 EV) ratio is verified empirically from
the pixels: undoing gamma 2.2 on well-exposed mid-tones gives a median linear ratio
of 8.08 (3.01 EV), extremely consistent across scenes.

Three reference-exposure modes (the middle frame is the HDR reference):

* ``low``       -> high, **low**, high   (short-exposure reference)
* ``high``      -> low,  **high**, low   (long-exposure reference)
* ``alternate`` -> globally alternating high/low/high/low... per absolute frame index

Within each scene a sliding window of 3 frames is taken: 0-1-2, 1-2-3, ..., 5-6-7.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import read_16bit_tif, read_hdr

# Empirically-verified exposures for Real-HDRV (short = unit 1, long = 8 -> 3 EV).
SHORT_EXP = 1.0
LONG_EXP = 8.0


class RealHDRV_Dataset(Dataset):
    def __init__(self, root_dir, nframes=3, nexps=2, ref_exposure="alternate"):
        assert nframes == 3, "Real-HDRV grouping is fixed to 3 frames."
        assert nexps == 2, "F2HDR supports 2 exposures only."
        assert ref_exposure in ("low", "high", "alternate"), \
            "ref_exposure must be 'low', 'high', or 'alternate'."

        self.root_dir = root_dir
        self.nframes = nframes
        self.ref_exposure = ref_exposure

        self.scene_list = sorted(os.listdir(self.root_dir))
        self.img_list = []   # each: list of 3 (ldr_path, 'low'/'high')
        self.hdr_list = []   # each: list of 3 hdr paths

        n_total = 8                                  # frames per scene
        frames = [f"frame_{i:02d}" for i in range(n_total)]
        group_starts = list(range(0, n_total - self.nframes + 1))  # 0..5

        # Per-group exposure template (low/high reference) ...
        if ref_exposure == "low":
            exposure_seq = ["high", "low", "high"]
        elif ref_exposure == "high":
            exposure_seq = ["low", "high", "low"]
        else:
            exposure_seq = None
        # ... or global per-frame template (alternate).
        alt_per_frame = (["high" if i % 2 == 0 else "low" for i in range(n_total)]
                         if ref_exposure == "alternate" else None)

        for scene in self.scene_list:
            scene_dir = os.path.join(self.root_dir, scene)
            input_low = os.path.join(scene_dir, "Input_0")   # short / low
            input_high = os.path.join(scene_dir, "Input_1")  # long  / high
            gt_dir = os.path.join(scene_dir, "GT")

            for s in group_starts:
                group = frames[s:s + self.nframes]
                img_paths, hdr_paths = [], []
                for i, frame in enumerate(group):
                    want = alt_per_frame[s + i] if ref_exposure == "alternate" else exposure_seq[i]
                    if want == "low":
                        img_paths.append((os.path.join(input_low, f"{frame}.tif"), "low"))
                    else:
                        img_paths.append((os.path.join(input_high, f"{frame}.tif"), "high"))
                    hdr_paths.append(os.path.join(gt_dir, f"GT_{frame}.hdr"))
                self.img_list.append(img_paths)
                self.hdr_list.append(hdr_paths)

        print("[%s] totaling %d samples (ref_exposure=%s)"
              % (self.__class__.__name__, len(self.img_list), ref_exposure))

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        hdr_paths = self.hdr_list[index]

        ldrs, expos = [], []
        for ldr_path, exposure_type in img_paths:
            # cv2-based reader handles LZW-compressed 16-bit TIFFs and returns RGB in [0,1].
            ldrs.append(read_16bit_tif(ldr_path))
            expos.append(SHORT_EXP if exposure_type == "low" else LONG_EXP)

        # Reference (middle) frame HDR is the target.
        hdr = read_hdr(hdr_paths[1])
        if hdr.max() > 1:
            hdr = hdr / hdr.max()
        hdr_tensor = torch.from_numpy(hdr.astype(np.float32).transpose(2, 0, 1))

        ldrs_tensor, expos_tensor = [], []
        for i in range(len(ldrs)):
            ldrs_tensor.append(torch.from_numpy(ldrs[i].astype(np.float32).transpose(2, 0, 1)))
            expos_tensor.append(torch.tensor(expos[i]))

        return {
            "hdr_path": hdr_paths[1].split("/")[-2] + "_" + hdr_paths[1].split("/")[-1],
            "hdr": hdr_tensor,
            "ldrs": ldrs_tensor,
            "expos": expos_tensor,
        }

    def __len__(self):
        return len(self.img_list)
