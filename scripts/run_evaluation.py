"""Compute Quantus metrics for every attribution method over the eval batch
"""

import argparse
from attributions import deconvnet, deeplift, gradcam, gradient_based, lrp, occlusion
from attributions.perturbation_based import kernel_shap, lime
from attributions.utils import slic_feature_mask_batch
from config import DEVICE, RESULTS_DIR
from data.loader import load_eval_batch
from evaluation.run_all import run_all_metrics
from models.vgg16 import load_vgg16
from visualization.radar_charts import plot_radar
from visualization.rank_bar import plot_rank_bar, rank_dataframe

METHOD_CATEGORIES = {
    "Perturbation-Based": ["DeconvNets", "IntegratedGradients", "InputXGradient", "Occlusion", "KernelShap", "LIME"],
    "Gradient-Based": ["Saliency", "GradCAM", "GradCAM++", "SmoothGrad"],
    "Decomposition-Based": ["DeepLift", "LRP", "CLRP"],
}

RADAR_SPECS = [
    ("Perturbation-Based", ["Robustness", "Randomisation"], ["#AB63FA", "#00CC96"], "Perturbation-Based Quantification"),
    ("Gradient-Based", ["Complexity", "Faithfulness"], ["#636EFA", "#EF553B"], "Gradient-Based Quantification"),
    ("Decomposition-Based", ["Localisation", "Faithfulness"], ["#FECB52", "#EF553B"], "Decomposition-Based Quantification"),
]


def build_explanations(model, x_batch, y_batch, feature_mask):
    explanations = {}

    explanations["DeconvNets"] = deconvnet.compute(model, x_batch, y_batch)
    explanations["IntegratedGradients"] = gradient_based.integrated_gradients(
        model, x_batch, y_batch, n_steps=40
    )
    explanations["InputXGradient"] = gradient_based.input_x_gradient(model, x_batch, y_batch)
    explanations["Occlusion"] = occlusion.compute(
        model, x_batch, y_batch, strides=(3, 15, 15), sliding_window_shapes=(3, 17, 17), baseline="zeros"
    )
    explanations["Saliency"] = gradient_based.saliency(model, x_batch, y_batch, smooth=False)
    explanations["GradCAM"] = gradcam.gradcam(model, x_batch)
    explanations["GradCAM++"] = gradcam.gradcam_plusplus(model, x_batch)
    explanations["SmoothGrad"] = gradcam.smoothgrad_gradcam_plusplus(model, x_batch)
    explanations["DeepLift"] = deeplift.compute(model, x_batch, y_batch)
    explanations["KernelShap"] = kernel_shap(model, x_batch, y_batch, feature_mask)
    explanations["LIME"] = lime(model, x_batch, y_batch, feature_mask)

    lrp.assign_lrp_rules(model, conv_cutoff=37)
    lrp_attributions, lrp_map = lrp.lrp(model, x_batch, y_batch)
    explanations["LRP"] = lrp_map
    explanations["CLRP"] = lrp.clrp(model, x_batch, lrp_attributions)

    return explanations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=str(RESULTS_DIR / "quantus_results.csv"))
    args = parser.parse_args()

    model = load_vgg16()
    x_batch, y_batch, s_batch = load_eval_batch(DEVICE)

    feature_mask = slic_feature_mask_batch(x_batch).to(DEVICE)
    explanations = build_explanations(model, x_batch, y_batch, feature_mask)

    df = run_all_metrics(model, x_batch, y_batch, s_batch, explanations)
    df.to_csv(args.csv_path)
    print(f"Saved metric results to {args.csv_path}")
    print(df)

    df_rank = rank_dataframe(df)
    plot_rank_bar(df_rank, save_path=str(RESULTS_DIR / "rank_summary.png"))

    df_t = df.T
    for category, metrics, colors, title in RADAR_SPECS:
        methods = [m for m in METHOD_CATEGORIES[category] if m in df_t.columns]
        plot_radar(
            df_t.loc[metrics, methods],
            metrics,
            title,
            colors,
            save_path=str(RESULTS_DIR / f"radar_{category.lower().replace('-', '_')}.png"),
        )


if __name__ == "__main__":
    main()
