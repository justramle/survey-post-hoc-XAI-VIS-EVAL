"""Helpers shared across attribution methods."""

import numpy as np
import torch
from skimage.segmentation import slic


def preprocess_attr(attributions):
    """[b, c, h, w] attributions -> normalized [b, h, w] maps averaged over channels."""
    if isinstance(attributions, torch.Tensor):
        attributions_np = attributions.cpu().detach().numpy()
    else:
        attributions_np = attributions

    attributions_np = np.transpose(attributions_np, (0, 2, 3, 1))
    attributions_np = np.mean(attributions_np, axis=3)
    attributions_np = (attributions_np.T / np.max(attributions_np, axis=(1, 2)).T).T
    return attributions_np


def slic_feature_mask(image_chw, n_segments=90, compactness=3, sigma=1):
    """Superpixel mask for a single [3, H, W] image, broadcast to 3 channels."""
    image_hwc = np.array(image_chw.permute(1, 2, 0))
    segments = slic(
        image_hwc, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1
    )
    mask = np.repeat(segments[None, None, :, :], 3, axis=1)
    return torch.tensor(mask, dtype=torch.long)


def slic_feature_mask_batch(x_batch, n_segments=100, compactness=10, sigma=1):
    """Superpixel masks for a [B, 3, H, W] batch, one channel (KernelShap/Lime broadcast it)."""
    masks = []
    for img in x_batch:
        img_hwc = img.permute(1, 2, 0).cpu().numpy()
        segments = slic(img_hwc, n_segments=n_segments, compactness=compactness, sigma=sigma)
        masks.append(torch.tensor(segments).unsqueeze(0))
    return torch.stack(masks).long()


def add_gaussian_noise(inputs, stdev_spread, n_samples):
    stdev = stdev_spread * (inputs.max() - inputs.min())
    return [inputs + torch.randn_like(inputs) * stdev for _ in range(n_samples)]
