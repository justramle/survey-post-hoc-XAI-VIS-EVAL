import numpy as np
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import KernelShap, Lime
from captum.attr._core.lime import get_exp_kernel_similarity_function
from .utils import preprocess_attr


def kernel_shap(model, inputs, target, feature_mask):
    attributions = KernelShap(model).attribute(inputs, target=target, feature_mask=feature_mask)
    return preprocess_attr(attributions)


def lime(model, inputs, target, feature_mask, n_samples=40, perturbations_per_eval=16):
    similarity_func = get_exp_kernel_similarity_function("euclidean", kernel_width=1000)
    lr_lime = Lime(
        model,
        interpretable_model=SkLearnLinearRegression(),
        similarity_func=similarity_func,
    )
    attributions = lr_lime.attribute(
        inputs,
        target=target,
        feature_mask=feature_mask,
        n_samples=n_samples,
        perturbations_per_eval=perturbations_per_eval,
        show_progress=True,
    )
    return preprocess_attr(attributions)


def threshold_low_relevance(attr_map, percentile=2):
    threshold = np.percentile(attr_map, percentile)
    return np.where(attr_map > threshold, attr_map, 0)
