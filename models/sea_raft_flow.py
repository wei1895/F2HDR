"""Frozen SEA-RAFT wrapper: [prev, cur, nxt] -> [flow(cur->prev), flow(cur->next)].

The backbone is frozen (requires_grad=False, forced eval, no_grad forward) and lives
under ``self.net`` so it can be excluded from checkpoints (see utils.checkpoint).
"""
import torch
import torch.nn as nn

from sea_raft import build_sea_raft


class FrozenSEARAFT(nn.Module):
    def __init__(self, config_path, ckpt_path, iters=None):
        super().__init__()
        self.net = build_sea_raft(config_path, ckpt_path)
        self.iters = iters if iters is not None else self.net.args.iters
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    def train(self, mode=True):
        """Keep the frozen backbone in eval mode regardless of parent's train/eval."""
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, ldrs):
        prev, cur, nxt = ldrs
        # SEA-RAFT expects inputs in the [0, 255] range.
        cur255, prev255, nxt255 = cur * 255.0, prev * 255.0, nxt * 255.0
        p_flow = self.net(cur255, prev255, iters=self.iters, test_mode=True)["final"]
        n_flow = self.net(cur255, nxt255, iters=self.iters, test_mode=True)["final"]
        return [p_flow, n_flow]
