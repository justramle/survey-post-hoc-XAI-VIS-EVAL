"""Generate a side-by-side attribution map comparison for a single image.

Usage (from the project root):
    python -m scripts.run_visualization --image data/assets/cat1.png
"""

import argparse

import numpy as np

from attributions import deconvnet, deeplift, gradcam, gradient_based, lrp, occlusion
from attributions.perturbation_based import kernel_shap, lime, threshold_low_relevance
from attributions.utils import slic_feature_mask
from config import DEVICE, RESULTS_DIR, SAMPLE_IMAGE_PATH
from data.loader import load_image, load_imagenet_labels, predict
from models.vgg16 import load_vgg16
from visualization.attribution_grid import plot_attribution_grid


def build_explanations(model, model_input, target, feature_mask):
    explanations = {}

    explanations["DeconvNets"] = deconvnet.compute(model, model_input, target)
    explanations["IntegratedGradients"] = gradient_based.integrated_gradients(
        model, model_input, target, smooth=True
    )
    explanations["InputXGradient"] = gradient_based.input_x_gradient(model, model_input, target)
    explanations["Occlusion"] = occlusion.compute(model, model_input, target)
    explanations["Saliency"] = gradient_based.saliency(model, model_input, target)
    explanations["GradCAM"] = gradcam.gradcam(model, model_input)
    explanations["GradCAM++"] = gradcam.gradcam_plusplus(model, model_input)
    explanations["SmoothGrad"] = gradcam.smoothgrad_gradcam_plusplus(model, model_input)
    explanations["DeepLift"] = deeplift.compute(model, model_input, target)

    ks_map = kernel_shap(model, model_input, target, feature_mask)
    explanations["KernelShap"] = threshold_low_relevance(ks_map)

    lime_map = lime(model, model_input, target, feature_mask)
    explanations["LIME"] = threshold_low_relevance(lime_map)

    lrp.assign_lrp_rules(model)
    lrp_attributions, lrp_map = lrp.lrp(model, model_input, target)
    explanations["LRP"] = lrp_map
    explanations["CLRP"] = lrp.clrp(model, model_input, lrp_attributions)

    return explanations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(SAMPLE_IMAGE_PATH))
    parser.add_argument("--save-path", default=str(RESULTS_DIR / "attribution_grid.png"))
    args = parser.parse_args()

    model = load_vgg16()
    labels = load_imagenet_labels()

    display_tensor, model_input = load_image(args.image)
    model_input = model_input.to(DEVICE)

    target, score, predicted_label = predict(model, model_input, labels)
    print(f"Predicted: {predicted_label} ({score:.4f})")

    feature_mask = slic_feature_mask(display_tensor).to(DEVICE)

    explanations = build_explanations(model, model_input, target, feature_mask)

    original_image = display_tensor.numpy().transpose(1, 2, 0)
    plot_attribution_grid(original_image, explanations, save_path=args.save_path)
    print(f"Saved attribution grid to {args.save_path}")


if __name__ == "__main__":
    main()
