import torch
from captum.attr import DeepLift

from .utils import preprocess_attr

def compute(model, inputs, target, baseline=None):
    baseline = torch.zeros_like(inputs) if baseline is None else baseline
    attributions = DeepLift(model).attribute(inputs, baseline, target=target)
    return preprocess_attr(attributions)
