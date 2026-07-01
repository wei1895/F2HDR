"""Shared network utilities: HDR conversions, warping, and conv building blocks."""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "OrderedDict", "activation", "conv_layer", "deconv_layer", "output_conv",
    "coords_grid", "backward_warp", "ldr_to_hdr", "adj_expo_ldr_to_ldr",
    "merge_hdr", "MergeHDRModule",
]


# ---------------------------------------------------------------------------
# Conv building blocks (used by the fusion U-Net)
# ---------------------------------------------------------------------------
def activation(afunc="LReLU", inplace=True):
    if afunc == "LReLU":
        return nn.LeakyReLU(0.1, inplace=inplace)
    if afunc == "LReLU02":
        return nn.LeakyReLU(0.2, inplace=inplace)
    if afunc == "ReLU":
        return nn.ReLU(inplace=inplace)
    if afunc == "Sigmoid":
        return nn.Sigmoid()
    if afunc == "Tanh":
        return nn.Tanh()
    raise ValueError("Unknown activation function: %s" % afunc)


def conv_layer(cin, cout, k=3, stride=1, pad=-1, dilation=1, afunc="LReLU",
               use_bn=False, bias=True, inplace=True):
    if not isinstance(pad, tuple):
        pad = pad if pad >= 0 else (k - 1) // 2
    block = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=bias, dilation=dilation)]
    if use_bn:
        block += [nn.BatchNorm2d(cout)]
    if afunc:
        block += [activation(afunc, inplace=inplace)]
    return nn.Sequential(*block)


def deconv_layer(cin, cout, afunc="LReLU", use_bn=False, bias=False):
    block = [nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=bias)]
    if use_bn:
        block += [nn.BatchNorm2d(cout)]
    block += [activation(afunc)]
    return nn.Sequential(*block)


def output_conv(cin, cout, k=1, stride=1, pad=0, bias=True):
    pad = (k - 1) // 2
    return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=bias))


# ---------------------------------------------------------------------------
# Warping / coordinate utilities
# ---------------------------------------------------------------------------
def coords_grid(b, h, w, device):
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(b, 1, 1, 1)


def backward_warp(img, flow, pad="zeros"):
    """Warp ``img`` toward the reference frame using ``flow`` (cur -> img direction)."""
    b, c, h, w = img.shape
    grid = coords_grid(b, h, w, device=img.device) + flow
    xgrid, ygrid = grid.split([1, 1], dim=1)
    xgrid = 2 * xgrid / (w - 1) - 1
    ygrid = 2 * ygrid / (h - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=1)
    return F.grid_sample(img, grid.permute(0, 2, 3, 1), mode="bilinear",
                         padding_mode=pad, align_corners=False)


# ---------------------------------------------------------------------------
# HDR / LDR radiance conversions
# ---------------------------------------------------------------------------
def ldr_to_hdr(ldr, expo, gamma=2.2):
    """Map an LDR frame to linear HDR radiance given its exposure."""
    ldr = ldr.clamp(0, 1)
    ldr = torch.pow(ldr, gamma)
    expo = expo.view(-1, 1, 1, 1)
    return ldr / expo


def adj_expo_ldr_to_ldr(ldr, c_exp, p_exp, gamma=2.2):
    """Re-expose an LDR frame from exposure ``c_exp`` to ``p_exp``."""
    gain = torch.pow(p_exp / c_exp, 1.0 / gamma).view(-1, 1, 1, 1)
    return (ldr * gain).clamp(0, 1)


# ---------------------------------------------------------------------------
# Weighted HDR merging
# ---------------------------------------------------------------------------
def merge_hdr(ws, hdrs):
    assert len(ws) == len(hdrs)
    w_sum = torch.stack(ws, 1).sum(1)
    ws = [w / (w_sum + 1e-8) for w in ws]
    hdr = ws[0] * hdrs[0]
    for i in range(1, len(ws)):
        hdr = hdr + ws[i] * hdrs[i]
    return hdr, ws


class MergeHDRModule(nn.Module):
    """Normalize fusion weights and produce the weighted-sum HDR radiance."""

    def forward(self, ws, hdrs):
        return merge_hdr(ws, hdrs)
