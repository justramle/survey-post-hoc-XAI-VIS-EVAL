import pandas as pd
from . import complexity, faithfulness, localisation, randomisation, robustness
from .metrics import run_metric


def run_all_metrics(model, x_batch, y_batch, s_batch, explanations):
    results = {
        "Robustness": run_metric(robustness.avg_sensitivity(), model, x_batch, y_batch, explanations),
        "Faithfulness": run_metric(
            faithfulness.sufficiency(), model, x_batch, y_batch, explanations
        ),
        "Localisation": run_metric(
            localisation.pointing_game(), model, x_batch, y_batch, explanations, s_batch=s_batch
        ),
        "Complexity": run_metric(complexity.sparseness(), model, x_batch, y_batch, explanations),
        "Randomisation": run_metric(
            randomisation.random_logit(), model, x_batch, y_batch, explanations
        ),
    }
    return pd.DataFrame(results)
