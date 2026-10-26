import matplotlib.pyplot as plt


def plot_attribution_grid(original_image, explanations, ncols=None, save_path=None, cmap="seismic"):
    n_panels = len(explanations) + 1
    ncols = ncols or n_panels
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = axes.flatten() if n_panels > 1 else [axes]

    axes[0].imshow(original_image, vmin=0.0, vmax=1.0)
    axes[0].set_title("Original Image", fontsize=11)

    for ax, (method, attr_map) in zip(axes[1:], explanations.items()):
        ax.imshow(attr_map.squeeze(), cmap=cmap, interpolation="bilinear", vmin=-1, vmax=1)
        ax.set_title(method, fontsize=11)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig
