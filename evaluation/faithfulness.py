import numpy as np
import quantus


def faithfulness_correlation():
    return quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=True,
        normalise=True,
        normalise_func=quantus.normalise_by_max,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    )


def sufficiency():
    return quantus.Sufficiency(
        threshold=0.9,
        distance_func="seuclidean",
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_by_max,
        return_aggregate=True,
        aggregate_func=np.mean,
        disable_warnings=True,
        display_progressbar=False,
    )
