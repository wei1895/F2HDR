"""Flow Adapter: a lightweight, trainable refiner on top of frozen SEA-RAFT flow."""
import torch
import torch.nn as nn

from .network_utils import backward_warp

DIV_FLOW = 20.0


def conv_prelu(cin, cout, k=3, stride=1, pad=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation, bias=True),
        nn.PReLU(cout),
    )


class ResDilatedBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=True),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=True),
        )
        self.act = nn.PReLU(c)

    def forward(self, x):
        return self.act(x + self.body(x))


class FlowAdapter(nn.Module):
    def __init__(self, c=64, dilations=(1, 2, 4, 8, 4, 2)):
        super().__init__()
        in_ch = 3 + 3 + 3 + 2 + 2  # warp(prev), cur, warp(next), flow_p/div, flow_n/div
        self.head = conv_prelu(in_ch, c)
        self.body = nn.Sequential(*[ResDilatedBlock(c, d) for d in dilations])
        self.fuse = conv_prelu(c, c)
        self.tail = nn.Conv2d(c, 4, 3, padding=1, bias=True)
        # Start as near-identity: small residual at init so we begin from SEA-RAFT flow.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, ldrs, coarse_flows):
        # ldrs: [prev, cur, nxt]; coarse_flows: [p_flow, n_flow] (cur->prev, cur->next).
        p_flow, n_flow = coarse_flows
        p_warp = backward_warp(ldrs[0], p_flow)
        n_warp = backward_warp(ldrs[2], n_flow)

        x = torch.cat([p_warp, ldrs[1], n_warp, p_flow / DIV_FLOW, n_flow / DIV_FLOW], dim=1)
        x = self.head(x)
        x = self.body(x)
        x = self.fuse(x)
        res = self.tail(x)

        flow_p = (p_flow + res[:, 0:2] * DIV_FLOW).clamp(-100, 100)
        flow_n = (n_flow + res[:, 2:4] * DIV_FLOW).clamp(-100, 100)
        return [flow_p, flow_n]
