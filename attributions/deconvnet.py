from captum.attr import Deconvolution
from .utils import preprocess_attr


def compute(model, inputs, target):
    attributions = Deconvolution(model).attribute(inputs, target=target)
    return preprocess_attr(attributions)
