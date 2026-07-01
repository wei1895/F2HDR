"""Evaluate F2HDR on HDR video benchmarks (2-exposure).

Supported datasets:
  * ``DeepHDRVideo``   -- real dynamic / static benchmark
  * ``CinematicVideo`` -- HDR_Synthetic_Test_Dataset
  * ``RealHDRV``       -- Real-HDRV test split (Test_Compack_8frames_50scenes)

Run::

    python test.py --dataset DeepHDRVideo --dataset_dir <dir> --pretrained_model <ckpt> [--save_results]
    python test.py --dataset RealHDRV     --dataset_dir <dir> --pretrained_model <ckpt> --ref_exp alternate
"""
import argparse
import logging
import os
import os.path as osp

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

import cv2
from dataset import Real_Benchmark_Dataset, RealHDRV_Dataset, Syn_Test_Dataset
from models import F2HDR
from models.network_utils import ldr_to_hdr as ldr_to_hdr_t
from utils.checkpoint import load_checkpoint
from utils.flow_viz import flow_to_image
from utils.utils import AverageMeter, InputPadder, save_hdr, tonemap


def get_args():
    parser = argparse.ArgumentParser(description="F2HDR evaluation (2-exposure)")
    parser.add_argument("--dataset", type=str, default="DeepHDRVideo",
                        choices=["DeepHDRVideo", "CinematicVideo", "RealHDRV"])
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--ref_exp", type=str, default="alternate",
                        choices=["low", "high", "alternate"],
                        help="Real-HDRV reference-exposure mode (ignored by other datasets)")
    parser.add_argument("--save_results", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="./test_results/f2hdr_2E")
    return parser.parse_args()


def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)


def save_flow(flow_preds, path):
    """Save the two refined flows (cur->prev | cur->next) as a colorized image."""
    imgs = []
    for flow in flow_preds:
        flow = torch.squeeze(flow).permute(1, 2, 0).cpu().numpy()
        imgs.append(flow_to_image(flow))
    concat = np.concatenate(imgs, axis=1)
    cv2.imwrite(path, concat[:, :, [2, 1, 0]].astype("uint8"))


@torch.no_grad()
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    setup_logging(os.path.join(args.save_dir, "test.log"))
    logging.info(">>> F2HDR evaluation on %s", args.dataset)
    logging.info("Weights: %s", args.pretrained_model)

    device = torch.device("cuda")
    model = F2HDR().to(device)
    load_checkpoint(args.pretrained_model, model, strict=False)
    model.eval()

    if args.dataset == "DeepHDRVideo":
        test_set = Real_Benchmark_Dataset(root_dir=args.dataset_dir, nframes=3, nexps=2)
    elif args.dataset == "RealHDRV":
        test_set = RealHDRV_Dataset(root_dir=args.dataset_dir, nframes=3, nexps=2,
                                    ref_exposure=args.ref_exp)
        logging.info("Real-HDRV reference-exposure mode: %s", args.ref_exp)
    else:
        test_set = Syn_Test_Dataset(root_dir=args.dataset_dir, nframes=3, nexps=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    meters = {k: AverageMeter() for k in ["psnrL", "psnrT", "ssimL", "ssimT"]}
    low = {k: AverageMeter() for k in ["psnrL", "psnrT"]}
    high = {k: AverageMeter() for k in ["psnrL", "psnrT"]}

    for idx, data in enumerate(test_loader):
        ldrs = [x.to(device) for x in data["ldrs"]]
        expos = [x.to(device) for x in data["expos"]]
        gt_hdr = data["hdr"]

        padder = InputPadder(ldrs[0].shape, divis_by=16)
        pad_ldrs = padder.pad(ldrs)
        pred_hdr, flow_preds = model(pad_ldrs, expos, test_mode=True)
        pred_hdr = padder.unpad(pred_hdr)
        flow_preds = [padder.unpad(f) for f in flow_preds]
        pred_hdr = torch.squeeze(pred_hdr.cpu()).numpy().astype(np.float32).transpose(1, 2, 0)

        cur_ldr = torch.squeeze(ldrs[1].cpu()).numpy().astype(np.float32).transpose(1, 2, 0)
        Y = 0.299 * cur_ldr[:, :, 0] + 0.587 * cur_ldr[:, :, 1] + 0.114 * cur_ldr[:, :, 2]
        Y = Y[:, :, None]
        mask = Y < 0.2 if expos[1] <= 1.0 else Y > 0.8
        cur_linear = ldr_to_hdr_t(ldrs[1], expos[1])
        cur_linear = torch.squeeze(cur_linear.cpu()).numpy().astype(np.float32).transpose(1, 2, 0)
        pred_hdr = (~mask) * cur_linear + mask * pred_hdr

        gt_hdr = torch.squeeze(gt_hdr).numpy().astype(np.float32).transpose(1, 2, 0)
        pred_tm, gt_tm = tonemap(pred_hdr), tonemap(gt_hdr)

        psnrL = psnr(gt_hdr, pred_hdr)
        psnrT = psnr(gt_tm, pred_tm)
        ssimL = ssim(gt_hdr, pred_hdr, channel_axis=2, data_range=gt_hdr.max() - gt_hdr.min())
        ssimT = ssim(gt_tm, pred_tm, channel_axis=2, data_range=gt_tm.max() - gt_tm.min())

        meters["psnrL"].update(psnrL); meters["psnrT"].update(psnrT)
        meters["ssimL"].update(ssimL); meters["ssimT"].update(ssimT)
        bucket = low if expos[1] <= 1.0 else high
        bucket["psnrL"].update(psnrL); bucket["psnrT"].update(psnrT)
        logging.info("[%d/%d] %s  PSNR-mu: %.4f  PSNR-l: %.4f  SSIM-mu: %.4f  SSIM-l: %.4f",
                     idx + 1, len(test_loader),
                     "Low " if expos[1] <= 1.0 else "High", psnrT, psnrL, ssimT, ssimL)

        if args.save_results:
            name = args.dataset_dir.rstrip("/").split("/")[-1]
            if args.dataset == "RealHDRV":
                name = "%s_%s" % (name, args.ref_exp)
            out_dir = os.path.join(args.save_dir, name, "hdr_output")
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "%d_pred.png" % (idx + 1)),
                        (pred_tm * 255.)[:, :, [2, 1, 0]].astype("uint8"))
            hdr_dir = os.path.join(args.save_dir, name, "hdr_format_output")
            os.makedirs(hdr_dir, exist_ok=True)
            save_hdr(os.path.join(hdr_dir, "%d_pred.hdr" % (idx + 1)), pred_hdr)

            flow_dir = os.path.join(args.save_dir, name, "flow_preds")
            os.makedirs(flow_dir, exist_ok=True)
            save_flow(flow_preds, os.path.join(flow_dir, "%d_flow.png" % (idx + 1)))

    logging.info("Low  Average PSNR-mu: %.4f  PSNR-l: %.4f", low["psnrT"].avg, low["psnrL"].avg)
    logging.info("High Average PSNR-mu: %.4f  PSNR-l: %.4f", high["psnrT"].avg, high["psnrL"].avg)
    logging.info("All  Average PSNR-mu: %.4f  PSNR-l: %.4f", meters["psnrT"].avg, meters["psnrL"].avg)
    logging.info("All  Average SSIM-mu: %.4f  SSIM-l: %.4f", meters["ssimT"].avg, meters["ssimL"].avg)
    logging.info(">>> Done")


if __name__ == "__main__":
    main()
