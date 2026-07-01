"""Train F2HDR (2-exposure, 3-frame HDR video reconstruction).

Trains on Vimeo-90K (synthetic alternating-exposure LDRs) and validates on the
DeepHDRVideo dynamic benchmark after every epoch. Only the trainable sub-modules are
optimized and saved; the SEA-RAFT backbone stays frozen. Run::

    python train.py --dataset_vimeo_dir <vimeo> --dataset_chen_val_dir <chen> --logdir <out>
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset import fetch_dataloader
from models import F2HDR, F2HDR_Loss
from models.network_utils import ldr_to_hdr as ldr_to_hdr_t
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.utils import (AverageMeter, InputPadder, adjust_learning_rate,
                         set_random_seed, tonemap)

def get_args():
    parser = argparse.ArgumentParser(
        description="F2HDR training (2-exposure)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_vimeo_dir", type=str,
                        default="./data/vimeo_septuplet",
                        help="Vimeo-90K septuplet directory (training)")
    parser.add_argument("--dataset_chen_val_dir", type=str,
                        default="./data/dynamic_RGB_data_2exp_release",
                        help="DeepHDRVideo dynamic benchmark directory (per-epoch validation)")
    parser.add_argument("--logdir", type=str, default="./experiments/f2hdr_2E",
                        help="output directory for logs and checkpoints")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_epochs", type=str, default="20,30,40:2",
                        help="epochs to decay LR : downscale rate")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def setup_logging(logdir):
    os.makedirs(logdir, exist_ok=True)
    log_file = os.path.join(logdir, "train.log")
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
    logging.info("Training log -> %s", log_file)


def train_one_epoch(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    batch_time, data_time = AverageMeter(), AverageMeter()
    end = time.time()
    for batch_idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        ldrs = [x.to(device) for x in batch["ldrs"]]
        expos = [x.to(device) for x in batch["expos"]]
        hdrs = [x.to(device) for x in batch["hdrs"]]
        flow_gts = [x.to(device) for x in batch["flow_gts"]]
        flow_mask = batch["flow_mask"].to(device)

        pred_hdr, flow_preds = model(ldrs, expos)
        loss = criterion(pred_hdr, hdrs, flow_preds, ldrs[1], flow_mask, flow_gts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t"
                "Time: {bt.val:.3f} ({bt.avg:.3f})\tData: {dt.val:.3f} ({dt.avg:.3f})".format(
                    epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                    loss.item(), bt=batch_time, dt=data_time))


@torch.no_grad()
def validate(args, model, device, val_loader, epoch):
    """Evaluate on the DeepHDRVideo dynamic benchmark (full-frame, poorly-exposed compositing)."""
    model.eval()
    meters = {k: AverageMeter() for k in ["psnrL", "psnrT", "ssimL", "ssimT"]}
    for batch in val_loader:
        ldrs = [x.to(device) for x in batch["ldrs"]]
        expos = [x.to(device) for x in batch["expos"]]
        gt_hdr = batch["hdr"]

        padder = InputPadder(ldrs[0].shape, divis_by=16)
        pad_ldrs = padder.pad(ldrs)
        pred_hdr, _ = model(pad_ldrs, expos, test_mode=True)
        pred_hdr = padder.unpad(pred_hdr)

        pred_hdr = torch.squeeze(pred_hdr.cpu()).numpy().astype(np.float32).transpose(1, 2, 0)
        cur_ldr = torch.squeeze(ldrs[1].cpu()).numpy().astype(np.float32).transpose(1, 2, 0)
        gt_hdr = torch.squeeze(gt_hdr.cpu()).numpy().astype(np.float32).transpose(1, 2, 0)

        # Trust the reference frame in well-exposed regions; use the network elsewhere.
        Y = 0.299 * cur_ldr[:, :, 0] + 0.587 * cur_ldr[:, :, 1] + 0.114 * cur_ldr[:, :, 2]
        Y = Y[:, :, None]
        mask = Y < 0.2 if expos[1] <= 1.0 else Y > 0.8
        cur_linear = ldr_to_hdr_t(ldrs[1], expos[1])
        cur_linear = torch.squeeze(cur_linear.cpu()).numpy().astype(np.float32).transpose(1, 2, 0)
        pred_hdr = (~mask) * cur_linear + mask * pred_hdr

        pred_tm, gt_tm = tonemap(pred_hdr), tonemap(gt_hdr)
        meters["psnrL"].update(psnr(gt_hdr, pred_hdr))
        meters["psnrT"].update(psnr(gt_tm, pred_tm))
        meters["ssimL"].update(ssim(gt_hdr, pred_hdr, channel_axis=2, data_range=gt_hdr.max() - gt_hdr.min()))
        meters["ssimT"].update(ssim(gt_tm, pred_tm, channel_axis=2, data_range=gt_tm.max() - gt_tm.min()))

    logging.info("[Val epoch %d] PSNR-mu: %.4f  PSNR-l: %.4f  SSIM-mu: %.4f  SSIM-l: %.4f",
                 epoch, meters["psnrT"].avg, meters["psnrL"].avg,
                 meters["ssimT"].avg, meters["ssimL"].avg)
    return meters["psnrT"].avg


def main():
    args = get_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    setup_logging(args.logdir)

    device = torch.device("cuda")
    model = F2HDR().to(device)
    criterion = F2HDR_Loss().to(device)
    # Only optimize trainable parameters (frozen SEA-RAFT excluded).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, strict=False)
        args.start_epoch = ckpt.get("epoch", 0)
        logging.info("Resumed from %s at epoch %d", args.resume, args.start_epoch)

    model = nn.DataParallel(model)
    train_loader, val_loader = fetch_dataloader(args)

    best_psnr = -1.0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train_one_epoch(args, model, device, train_loader, optimizer, epoch, criterion)
        val_psnr = validate(args, model, device, val_loader, epoch)

        save_checkpoint(os.path.join(args.logdir, "checkpoint_%d.pth" % epoch),
                        model, optimizer, epoch=epoch + 1)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(os.path.join(args.logdir, "best.pth"),
                            model, optimizer, epoch=epoch + 1,
                            extra={"val_psnr_mu": val_psnr})
            logging.info("New best PSNR-mu: %.4f (epoch %d)", best_psnr, epoch)


if __name__ == "__main__":
    main()
