"""Stage-2 refinement network (MSRF) for F2HDR."""
import torch
import torch.nn as nn

from .network_utils import backward_warp

DIV_FLOW = 20.0


def conv_prelu(cin, cout, k=3, s=1, p=1, d=1, g=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, dilation=d, groups=g, bias=bias),
        nn.PReLU(cout),
    )


class SEGate(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, c, r=8):
        super().__init__()
        mid = max(1, c // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, mid, 1, bias=True), nn.PReLU(mid),
            nn.Conv2d(mid, c, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class MSRFBlock(nn.Module):
    """Multi-scale residual fusion block: parallel dilated + depthwise branches, SE, residual."""

    def __init__(self, c, inner=None):
        super().__init__()
        inner = inner or c // 2
        self.b1 = conv_prelu(c, inner, k=3, p=1, d=1)
        self.b2 = conv_prelu(c, inner, k=3, p=2, d=2)
        self.b3 = conv_prelu(c, inner, k=3, p=4, d=4)
        self.dw = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=5, stride=1, padding=2, groups=c, bias=True), nn.PReLU(c),
            nn.Conv2d(c, inner, kernel_size=1, stride=1, padding=0, bias=True), nn.PReLU(inner),
        )
        self.merge = nn.Conv2d(inner * 4, c, kernel_size=1, bias=True)
        self.se = SEGate(c)
        self.out = nn.Sequential(nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True), nn.PReLU(c))

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.dw(x)], dim=1)
        y = self.merge(y)
        y = self.se(y)
        y = self.out(y)
        return x + y


class ModalityEncoder(nn.Module):
    """Encode HDR and LDR streams of a frame separately, then fuse to a frame feature."""

    def __init__(self, in_hdr=3, in_ldr=3, c=48):
        super().__init__()
        self.hdr_enc = nn.Sequential(conv_prelu(in_hdr, c), conv_prelu(c, c))
        self.ldr_enc = nn.Sequential(conv_prelu(in_ldr, c), conv_prelu(c, c))
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=True), nn.PReLU(c))

    def forward(self, x6):
        hdr, ldr = torch.split(x6, [3, 3], dim=1)
        fh = self.hdr_enc(hdr)
        fl = self.ldr_enc(ldr)
        return self.fuse(torch.cat([fh, fl], dim=1))


class FlowMaskGate(nn.Module):
    """Per-pixel neighbor gate from |flow| and the motion mask; output in [0, 1]."""

    def __init__(self, in_ch=3, c=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1, bias=True), nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1, bias=True), nn.PReLU(c),
            nn.Conv2d(c, 1, 1, 1, 0, bias=True), nn.Sigmoid(),
        )

    def forward(self, flow, mask):
        mag = torch.sqrt(torch.clamp(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2, min=1e-12)) / DIV_FLOW
        x = torch.cat([mag, mask, torch.ones_like(mask)], dim=1)
        return self.net(x)


class RefineNet(nn.Module):
    def __init__(self, c=48, num_msrf=3):
        super().__init__()
        self.enc = ModalityEncoder(c=c)
        self.gate0 = FlowMaskGate(in_ch=3)  # prev gate from |flow_p|, mask_p
        self.gate2 = FlowMaskGate(in_ch=3)  # next gate from |flow_n|, mask_n
        self.adapt_in = nn.Conv2d(c * 3, c, 1, bias=True)
        self.msrf = nn.Sequential(*[MSRFBlock(c) for _ in range(num_msrf)])
        self.cond = nn.Sequential(
            conv_prelu(3, c // 2), nn.Conv2d(c // 2, c, 3, 1, 1, bias=True), nn.PReLU(c))
        self.fuse_cond = nn.Sequential(nn.Conv2d(c * 2, c, 1, bias=True), nn.PReLU(c))
        self.head = nn.Sequential(conv_prelu(c, c), nn.Conv2d(c, 3, 3, 1, 1, bias=True))

    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        f0 = self.enc(img0_c)
        f1 = self.enc(img1_c)
        f2 = self.enc(img2_c)

        f0w = backward_warp(f0, flow0)
        f2w = backward_warp(f2, flow2)

        g0 = self.gate0(flow0, mask0)
        g2 = self.gate2(flow2, mask2)

        f_nei = g0 * f0w + g2 * f2w
        x = torch.cat([f_nei, f1, f_nei - f1], dim=1)
        x = self.adapt_in(x)
        x = self.msrf(x)

        cnd = self.cond(img_hdr_m)
        x = self.fuse_cond(torch.cat([x, cnd], dim=1))

        res = self.head(x)
        return torch.clamp(img_hdr_m + res, 0, 1)
