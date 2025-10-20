
import torch
from torch.utils.data import Dataset

class SyntheticImageDataset(Dataset):
    """Synthetic (N, C, H, W) dataset with integer labels for quick benchmarks."""
    def __init__(self, num_samples=50000, num_classes=10, image_size=(3, 32, 32), seed=42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand((num_samples, *image_size), generator=g)
        self.y = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
