import torch
from captum.attr import Occlusion

from .utils import preprocess_attr


def compute(
    model,
    inputs,
    target,
    strides=(3, 5, 5),
    sliding_window_shapes=(3, 12, 12),
    baseline="mean",
):
    if baseline == "mean":
        baselines = inputs.mean(dim=(2, 3), keepdim=True)
    else:
        baselines = torch.zeros_like(inputs)

    attributions = Occlusion(model).attribute(
        inputs,
        strides=strides,
        target=target,
        sliding_window_shapes=sliding_window_shapes,
        baselines=baselines,
    )
    return preprocess_attr(attributions)
