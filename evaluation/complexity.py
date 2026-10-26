import numpy as np
import quantus


def sparseness():
    return quantus.Sparseness(
        abs=True,
        normalise=True,
        normalise_func=quantus.normalise_by_max,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    )
