"""F2HDR training loss: mu-law reconstruction + HDR alignment + optional flow supervision."""
import math

import torch
import torch.nn as nn

from .network_utils import backward_warp


def tonemap(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)


class F2HDR_Loss(nn.Module):
    def __init__(self, mu=5000, align_weight=0.5, flow_weight=0.001):
        super().__init__()
        self.mu = mu
        self.align_weight = align_weight
        self.flow_weight = flow_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, hdrs, flow_preds, cur_ldr, flow_mask, flow_gts):
        gt = hdrs[1]
        mu_pred = tonemap(pred, self.mu)
        mu_gt = tonemap(gt, self.mu)
        loss = self.l1(mu_pred, mu_gt)

        # Alignment loss in poorly-exposed regions of the reference frame.
        Y = 0.299 * cur_ldr[:, 0] + 0.587 * cur_ldr[:, 1] + 0.114 * cur_ldr[:, 2]
        Y = Y[:, None]
        mask = (Y > 0.8) | (Y < 0.2)
        mask = mask.repeat(1, 3, 1, 1)

        p_flow, n_flow = flow_preds
        if mask.sum() > 0:
            mu_p_warp_hdr = tonemap(backward_warp(hdrs[0], p_flow), self.mu)
            mu_n_warp_hdr = tonemap(backward_warp(hdrs[2], n_flow), self.mu)
            p_align_loss = self.l1(mu_p_warp_hdr[mask], mu_gt[mask])
            n_align_loss = self.l1(mu_n_warp_hdr[mask], mu_gt[mask])
            loss = loss + self.align_weight * (p_align_loss + n_align_loss)

        # Flow supervision (only where GT flow is valid).
        b, c, h, w = p_flow.shape
        flow_mask = flow_mask[:, None, None, None].repeat(1, 2, h, w) > 0.5
        if flow_mask.sum() > 0:
            p_flow_loss = self.l1(p_flow[flow_mask], flow_gts[0][flow_mask])
            n_flow_loss = self.l1(n_flow[flow_mask], flow_gts[1][flow_mask])
            loss = loss + self.flow_weight * (p_flow_loss + n_flow_loss)

        return loss
