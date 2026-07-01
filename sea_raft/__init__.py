"""Self-contained SEA-RAFT optical flow sub-package.

This is a trimmed, dependency-light port of the official SEA-RAFT
(https://github.com/princeton-vl/SEA-RAFT) ``core/`` directory. It exists so F2HDR
can use SEA-RAFT as a **frozen** coarse optical-flow estimator from source code +
local weights, without adding ``ptlflow`` / ``huggingface_hub`` third-party
dependencies.

Typical usage::

    from sea_raft import build_sea_raft
    model = build_sea_raft("configs/sea_raft_spring_m.json",
                           "pretrained_models/sea_raft/Tartan-C-T-TSKH-spring540x960-M.pth")

The returned module follows the SEA-RAFT API: ``model(image1, image2, iters=N,
test_mode=True)`` with images in the [0, 255] range.
"""
import argparse
import json

import torch

from .raft import RAFT

__all__ = ["RAFT", "load_config", "build_sea_raft"]


def load_config(json_path):
    """Load a SEA-RAFT JSON config into an ``argparse.Namespace``."""
    with open(json_path, "r") as f:
        data = json.load(f)
    args = argparse.Namespace()
    for key, value in data.items():
        setattr(args, key, value)
    return args


def build_sea_raft(config_path, ckpt_path=None, map_location="cpu"):
    """Construct a SEA-RAFT model and optionally load a checkpoint.

    Args:
        config_path: path to a SEA-RAFT JSON config (e.g. ``sea_raft_spring_m.json``).
        ckpt_path: optional path to a ``.pth`` state_dict. The official spring-M
            checkpoint loads with zero missing / unexpected keys.
        map_location: device for loading the checkpoint.

    Returns:
        The :class:`RAFT` model (in eval mode if a checkpoint was loaded).
    """
    args = load_config(config_path)
    model = RAFT(args)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[sea_raft] loaded {ckpt_path} with "
                  f"{len(missing)} missing / {len(unexpected)} unexpected keys")
        model.eval()
    return model
