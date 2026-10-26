import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from .utils import add_gaussian_noise


def _target_layer(model):
    return [model.features[-1]]


def gradcam(model, inputs):
    cam = GradCAM(model=model, target_layers=_target_layer(model))
    return cam(input_tensor=inputs)


def gradcam_plusplus(model, inputs):
    cam = GradCAMPlusPlus(model=model, target_layers=_target_layer(model))
    return cam(input_tensor=inputs)


def smoothgrad_gradcam_plusplus(model, inputs, stdev_spread=0.15, n_samples=50):
    cam = GradCAMPlusPlus(model=model, target_layers=_target_layer(model))
    noisy_inputs = add_gaussian_noise(inputs, stdev_spread, n_samples)
    cams = [cam(input_tensor=noisy_input) for noisy_input in noisy_inputs]
    return np.mean(cams, axis=0)
