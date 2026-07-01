"""Physical motion modeling -> soft motion mask (F2HDR)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

DIV_FLOW = 20.0  # flow normalization for the weight-predictor input (matches adapter)


class PhysicalMotionMask(nn.Module):
    def __init__(self, scales=(1, 2, 4), nbins=64, eps=1e-6):
        super().__init__()
        self.scales = tuple(scales)
        self.nbins = nbins
        self.eps = eps

        # Predicts the four fusion weights (w_t, w_d, w_c, w_s) from the flow.
        self.weight_net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.PReLU(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.PReLU(32),
            nn.Conv2d(32, 4, 1),
        )

        # Fixed Sobel kernels (registered as buffers so they move with .to(device)).
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad(self, comp):
        gx = F.conv2d(comp, self.sobel_x, padding=1)
        gy = F.conv2d(comp, self.sobel_y, padding=1)
        return gx, gy

    def _minmax(self, x):
        b = x.shape[0]
        xf = x.view(b, -1)
        mn = xf.min(dim=1)[0].view(b, 1, 1, 1)
        mx = xf.max(dim=1)[0].view(b, 1, 1, 1)
        return ((x - mn) / (mx - mn + self.eps)).clamp(0.0, 1.0)

    def _multi_scale_contrast(self, energy):
        contrasts = []
        for s in self.scales:
            center = F.avg_pool2d(energy, kernel_size=2 * s + 1, stride=1, padding=s)
            surround = F.avg_pool2d(energy, kernel_size=4 * s + 1, stride=1, padding=2 * s)
            contrasts.append((center - surround).abs())
        s_multi = torch.stack(contrasts, dim=0).mean(dim=0)
        return self._minmax(s_multi)

    @torch.no_grad()
    def _otsu_threshold(self, x):
        """Vectorized Otsu threshold per image on a [0, 1] energy map (graph-detached)."""
        b = x.shape[0]
        nbins = self.nbins
        xf = x.view(b, -1)
        idx = (xf * nbins).long().clamp(0, nbins - 1)
        hist = torch.zeros(b, nbins, device=x.device)
        hist.scatter_add_(1, idx, torch.ones_like(xf))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)

        centers = (torch.arange(nbins, device=x.device) + 0.5) / nbins
        omega = hist.cumsum(dim=1)
        mu = (hist * centers).cumsum(dim=1)
        mu_t = mu[:, -1:].clone()
        denom = (omega * (1.0 - omega)).clamp_min(1e-6)
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        best = sigma_b2.argmax(dim=1)
        return centers[best].view(b, 1, 1, 1)

    def forward(self, flow):
        """flow: (B, 2, H, W) -> soft mask (B, 1, H, W) in [0, 1]."""
        u = flow[:, 0:1]
        v = flow[:, 1:2]

        u_x, u_y = self._grad(u)
        v_x, v_y = self._grad(v)

        translation = torch.sqrt(u * u + v * v + self.eps)   # ||f||_2
        divergence = (u_x + v_y).abs()                       # |div(f)|
        curl = (v_x - u_y).abs()                             # |curl(f)|
        shear = (0.5 * (u_y + v_x)).abs()                    # |S|
        components = torch.cat([translation, divergence, curl, shear], dim=1)

        weights = F.softplus(self.weight_net(flow / DIV_FLOW)) + 1e-3
        energy = (weights * components).sum(dim=1, keepdim=True) / \
                 (weights.sum(dim=1, keepdim=True) + self.eps)

        s_multi = self._multi_scale_contrast(energy)
        energy = energy * (1.0 + 2.0 * s_multi)

        energy = self._minmax(energy)
        tau = self._otsu_threshold(energy)
        mask = 0.5 * (1.0 + torch.tanh(8.0 * (energy - tau)))
        return mask
