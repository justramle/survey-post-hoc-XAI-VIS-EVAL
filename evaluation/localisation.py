import numpy as np
import quantus


def relevance_rank_accuracy():
    return quantus.RelevanceRankAccuracy(
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_by_max,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    )


def pointing_game():
    return quantus.PointingGame(
        weighted=True,
        abs=True,
        normalise=True,
        normalise_func=quantus.functions.normalise_func.normalise_by_max,
        return_aggregate=True,
        aggregate_func=np.mean,
        default_plot_func=None,
        disable_warnings=True,
        display_progressbar=False,
    )


def relevance_mass_accuracy():
    return quantus.RelevanceMassAccuracy(
        abs=True,
        normalise=True,
        normalise_func=quantus.normalise_func.normalise_by_max,
        return_aggregate=False,
        aggregate_func=np.mean,
        disable_warnings=True,
    )
