import torch
from torchvision.utils import save_image
import numpy as np

def batch_mse_psnr(x, y, max_val=255.0, avg_over_batch=False):
    """Compute MSE and PSNR between two image batches."""
    # Manually add a batch dimension if input is 1D.
    if len(x.shape) == 1:
        x = x.view(-1, 1)
    if len(y.shape) == 1:
        y = y.view(-1, 1)

    x = x.float()
    y = y.float()

    # Compute squared difference
    squared_diff = (x - y) ** 2
    axes_except_batch = list(range(1, len(squared_diff.shape)))

    # Compute MSE per image in the batch
    mses = torch.mean(squared_diff, dim=axes_except_batch)  # per image

    # Compute PSNR
    psnrs = -10 * (torch.log10(mses) - 2 * torch.log10(torch.tensor(max_val)))

    if avg_over_batch:
        # Note that we average per-image PNSRs, following image compression convention.
        return torch.mean(mses), torch.mean(psnrs)
    else:
        return mses, psnrs

def mse_to_psnr(mse, max_val):
  return -10 * (np.log10(mse) - 2 * np.log10(max_val))