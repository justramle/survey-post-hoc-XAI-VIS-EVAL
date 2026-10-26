"""Loading the single demo image and the Quantus evaluation batch."""

import json

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import (
    IMAGE_SIZE,
    IMAGENET_LABELS_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    S_BATCH_PATH,
    X_BATCH_PATH,
    Y_BATCH_PATH,
)

_resize_to_tensor = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
)
_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_imagenet_labels():
    with open(IMAGENET_LABELS_PATH) as f:
        return json.load(f)


def load_image(image_path):
    """Returns (display_tensor [3,H,W] in [0,1], model_input [1,3,H,W] normalized)."""
    img = Image.open(image_path).convert("RGB")
    display_tensor = _resize_to_tensor(img)
    model_input = _normalize(display_tensor).unsqueeze(0)
    return display_tensor, model_input


def predict(model, model_input, labels):
    output = F.softmax(model(model_input), dim=1)
    score, label_idx = torch.topk(output, 1)
    label_idx = label_idx.squeeze()
    predicted_label = labels[str(label_idx.item())]
    return label_idx, score.squeeze().item(), predicted_label


def load_eval_batch(device):
    """Returns (x_batch, y_batch, s_batch) used by the Quantus metrics."""
    x_batch = torch.load(X_BATCH_PATH, map_location=device)
    y_batch = torch.load(Y_BATCH_PATH, map_location=device)
    s_batch = torch.load(S_BATCH_PATH, map_location=device).unsqueeze(dim=1)
    return x_batch, y_batch, s_batch
