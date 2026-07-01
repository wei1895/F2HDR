"""Dataset factory for F2HDR (2-exposure)."""
from torch.utils.data import DataLoader

from .real_benchmark_dataset import Real_Benchmark_Dataset
from .real_hdrv_dataset import RealHDRV_Dataset
from .syn_test_dataset import Syn_Test_Dataset
from .syn_vimeo_dataset import Syn_Vimeo_Dataset
from .tog13_online_align_dataset import TOG13_online_align_Dataset

__all__ = [
    "Syn_Vimeo_Dataset", "Real_Benchmark_Dataset", "RealHDRV_Dataset",
    "Syn_Test_Dataset", "TOG13_online_align_Dataset", "fetch_dataloader",
]


def fetch_dataloader(args):
    """Build the (Vimeo train, Chen-dynamic val) dataloaders used by ``train.py``."""
    train_set = Syn_Vimeo_Dataset(root_dir=args.dataset_vimeo_dir, nframes=3, nexps=2, is_training=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_set = Real_Benchmark_Dataset(root_dir=args.dataset_chen_val_dir, nframes=3, nexps=2)
    val_loader = DataLoader(
        val_set, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader
