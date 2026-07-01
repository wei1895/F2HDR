"""F2HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling."""
import os

import torch
import torch.nn as nn

from .flow_adapter import FlowAdapter
from .fusion import Fusion_Net
from .motion_mask import PhysicalMotionMask
from .network_utils import backward_warp, ldr_to_hdr
from .refine import RefineNet
from .sea_raft_flow import FrozenSEARAFT

# Default locations for the bundled SEA-RAFT config + weights (repo-relative).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEA_RAFT_CFG = os.path.join(_REPO_ROOT, "configs", "sea_raft_spring_m.json")
DEFAULT_SEA_RAFT_CKPT = os.path.join(
    _REPO_ROOT, "pretrained_models", "sea_raft", "Tartan-C-T-TSKH-spring540x960-M.pth")


def cur_tone_perturb(cur, test_mode, d=0.7):
    if test_mode:
        return cur
    b, c, h, w = cur.shape
    gamma_aug = torch.exp(torch.rand(b, 3, 1, 1, device=cur.device) * 2 * d - d)
    return torch.pow(cur, 1.0 / gamma_aug)


def prepare_fusion_inputs(ldrs, pt_cur, expos, flows):
    prev, cur, nxt = ldrs
    p_exp, c_exp, n_exp = expos
    p_flow, n_flow = flows

    p_warp = backward_warp(prev, p_flow)
    n_warp = backward_warp(nxt, n_flow)
    p_warp_hdr = ldr_to_hdr(p_warp, p_exp)
    n_warp_hdr = ldr_to_hdr(n_warp, n_exp)
    c_hdr = ldr_to_hdr(cur, c_exp)
    p_hdr = ldr_to_hdr(prev, p_exp)
    n_hdr = ldr_to_hdr(nxt, n_exp)
    pt_c_hdr = ldr_to_hdr(pt_cur, c_exp)

    hdrs = [pt_c_hdr, p_warp_hdr, n_warp_hdr, p_hdr, n_hdr]
    ldr_list = [pt_cur, p_warp, n_warp, prev, nxt]
    fusion_in = torch.cat(hdrs + ldr_list, dim=1)            # (B, 30, H, W)
    fusion_hdrs = [c_hdr, p_warp_hdr, n_warp_hdr, p_hdr, n_hdr]
    return fusion_in, fusion_hdrs


def prepare_refine_inputs(ldrs, expos, flows, masks, pt_cur):
    prev_ldr, cur_ldr, nxt_ldr = ldrs
    p_exp, c_exp, n_exp = expos
    p_flow, n_flow = flows
    mask0, mask2 = masks

    c_hdr = ldr_to_hdr(cur_ldr, c_exp)
    p_hdr = ldr_to_hdr(prev_ldr, p_exp)
    n_hdr = ldr_to_hdr(nxt_ldr, n_exp)
    pt_c_hdr = ldr_to_hdr(pt_cur, c_exp)

    img0_cat = torch.cat((p_hdr, prev_ldr), dim=1)
    img1_cat = torch.cat((pt_c_hdr, pt_cur), dim=1)
    img2_cat = torch.cat((n_hdr, nxt_ldr), dim=1)
    return img0_cat, img1_cat, img2_cat, p_flow, n_flow, mask0, mask2


class F2HDR(nn.Module):
    def __init__(self, sea_raft_cfg=DEFAULT_SEA_RAFT_CFG, sea_raft_ckpt=DEFAULT_SEA_RAFT_CKPT,
                 flow_iters=None):
        super().__init__()
        # Frozen coarse optical-flow backbone.
        self.coarse_flow_net = FrozenSEARAFT(sea_raft_cfg, sea_raft_ckpt, iters=flow_iters)
        # Trainable components.
        self.flow_adapter = FlowAdapter()
        self.motion_mask = PhysicalMotionMask()
        self.fusion_net = Fusion_Net(c_in=30, c_out=5, c_mid=128)
        self.refine_net = RefineNet()

    def forward(self, ldrs, expos, test_mode=False):
        prev, cur, nxt = ldrs
        pt_cur = cur_tone_perturb(cur, test_mode)

        # 1) frozen coarse flow -> 2) learned refinement
        coarse_flows = self.coarse_flow_net([prev, pt_cur, nxt])
        flow_preds = self.flow_adapter([prev, pt_cur, nxt], coarse_flows)

        # 3) physical motion masks from the refined flows
        masks = [self.motion_mask(flow_preds[0]), self.motion_mask(flow_preds[1])]

        # 4) stage-1 fusion -> coarse HDR
        fusion_in, fusion_hdrs = prepare_fusion_inputs(ldrs, pt_cur, expos, flow_preds)
        coarse_hdr = self.fusion_net(fusion_in, fusion_hdrs)

        # 5) stage-2 refinement -> final HDR
        img0, img1, img2, p_flow, n_flow, m0, m2 = prepare_refine_inputs(
            ldrs, expos, flow_preds, masks, pt_cur)
        pred_hdr = self.refine_net(img0, img1, img2, p_flow, n_flow, m0, m2, coarse_hdr)

        return pred_hdr, flow_preds

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
