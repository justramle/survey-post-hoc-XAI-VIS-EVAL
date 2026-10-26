from captum.attr import IntegratedGradients, InputXGradient, NoiseTunnel, Saliency
from .utils import preprocess_attr


def integrated_gradients(model, inputs, target, n_steps=50, smooth=False, nt_samples=10):
    ig = IntegratedGradients(model)
    if smooth:
        attributions = NoiseTunnel(ig).attribute(
            inputs, target=target, nt_type="smoothgrad", nt_samples=nt_samples
        )
    else:
        attributions = ig.attribute(inputs, target=target, n_steps=n_steps)
    return preprocess_attr(attributions)


def input_x_gradient(model, inputs, target, smooth=False, nt_samples=10):
    in_grad = InputXGradient(model)
    if smooth:
        attributions = NoiseTunnel(in_grad).attribute(
            inputs, target=target, nt_type="smoothgrad", nt_samples=nt_samples
        )
    else:
        attributions = in_grad.attribute(inputs, target=target)
    return preprocess_attr(attributions)


def saliency(model, inputs, target, smooth=True, nt_samples=15):
    sal = Saliency(model)
    if smooth:
        attributions = NoiseTunnel(sal).attribute(
            inputs, target=target, nt_type="smoothgrad", nt_samples=nt_samples
        )
    else:
        attributions = sal.attribute(inputs, target=target, abs=True)
    return preprocess_attr(attributions)
