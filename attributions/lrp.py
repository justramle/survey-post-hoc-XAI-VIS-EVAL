import torch
from captum.attr import LRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule

from .utils import preprocess_attr


def assign_lrp_rules(model, conv_cutoff=31):
    layers = list(model._modules["features"]) + list(model._modules["classifier"])
    for idx, layer in enumerate(layers[1:], start=1):
        if idx <= conv_cutoff:
            setattr(layer, "rule", Alpha1_Beta0_Rule())
        else:
            setattr(layer, "rule", EpsilonRule(epsilon=0))
    return model


def lrp(model, inputs, target):
    attributions = LRP(model).attribute(inputs, target=target)
    return attributions, preprocess_attr(attributions)


def clrp(model, inputs, lrp_attributions, contrast_scale=5):
    """Contrastive LRP: positive-class relevance minus scaled second-most-likely-class relevance."""
    probs = torch.softmax(model(inputs), dim=1)
    _, top2_classes = torch.topk(probs, 2, dim=1)
    contrastive_class = top2_classes[:, 1]

    attributions_neg = LRP(model).attribute(inputs, target=contrastive_class)
    attributions_clrp = lrp_attributions - contrast_scale * attributions_neg
    return preprocess_attr(attributions_clrp)
