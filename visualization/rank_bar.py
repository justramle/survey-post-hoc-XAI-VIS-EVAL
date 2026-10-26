import matplotlib.pyplot as plt
import seaborn as sns


def rank_dataframe(df):
    """Rank methods per metric. Robustness and Complexity: lower is better -> ascending rank.
    Faithfulness, Localisation, Randomisation: higher is better -> descending rank."""
    higher_is_better = [c for c in df.columns if c not in ("Robustness", "Complexity")]
    df_rank = df[higher_is_better].rank(ascending=False)
    df_rank["Robustness"] = df["Robustness"].rank(ascending=True)
    df_rank["Complexity"] = df["Complexity"].rank(ascending=True)
    return df_rank


def plot_rank_bar(df_rank, save_path=None):
    df_long = df_rank.reset_index().melt(id_vars="index", var_name="Metric", value_name="Rank")

    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_long, x="Metric", y="Rank", hue="index")
    plt.title("Summary of All Explainer Quantification", fontsize=18)
    plt.xlabel("Metric")
    plt.ylabel("Rank")
    plt.legend(title="Method", bbox_to_anchor=(1.01, 0.7), facecolor="white", frameon=False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()
