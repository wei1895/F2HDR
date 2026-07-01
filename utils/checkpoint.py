"""Checkpoint helpers that exclude the frozen SEA-RAFT backbone.

The frozen SEA-RAFT weights (~19.7M params, ~79 MB) never change during training, so
saving them in every checkpoint is wasteful. These helpers persist only the trainable
sub-modules (flow adapter, motion-mask predictor, fusion, refine) -- roughly 1.5M
params / ~6 MB. At load time the SEA-RAFT weights are restored from their bundled
``.pth`` (as the model is constructed) and only the trainable parts come from the
checkpoint.
"""
import torch
import torch.nn as nn

# Parameter-name prefix (after stripping any DataParallel ``module.``) under which the
# frozen SEA-RAFT backbone lives in :class:`models.f2hdr.F2HDR`.
FROZEN_PREFIX = "coarse_flow_net."


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def trainable_state_dict(model):
    """Return the model state_dict with the frozen SEA-RAFT entries removed."""
    net = _unwrap(model)
    full = net.state_dict()
    return {k: v for k, v in full.items() if not k.startswith(FROZEN_PREFIX)}


def save_checkpoint(path, model, optimizer=None, epoch=None, extra=None):
    """Save a checkpoint holding only the trainable parameters.

    Args:
        path: destination ``.pth`` path.
        model: the F2HDR model (optionally wrapped in ``DataParallel``).
        optimizer: optional optimizer whose state to store.
        epoch: optional epoch index to store.
        extra: optional dict of additional fields to merge in.
    """
    ckpt = {"state_dict": trainable_state_dict(model)}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True):
    """Load trainable parameters from a checkpoint into ``model``.

    The frozen SEA-RAFT backbone is left untouched (it is already initialized from its
    bundled weights when the model is constructed). Returns the loaded checkpoint dict.
    """
    net = _unwrap(model)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Tolerate checkpoints that were saved with a ``module.`` prefix.
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = net.load_state_dict(state, strict=False)
    # The only acceptable "missing" keys are the frozen SEA-RAFT ones.
    real_missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
    if strict and (real_missing or unexpected):
        raise RuntimeError(
            f"Checkpoint mismatch. Missing(trainable)={real_missing}, Unexpected={unexpected}")
    if real_missing or unexpected:
        print(f"[checkpoint] loaded {path} with "
              f"{len(real_missing)} missing(trainable) / {len(unexpected)} unexpected keys")

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
