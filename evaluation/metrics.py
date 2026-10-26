import numpy as np
import quantus


def run_metric(metric, model, x_batch, y_batch, explanations, s_batch=None):
    scores = {}
    for method, attr in explanations.items():
        kwargs = dict(
            model=model,
            x_batch=x_batch.cpu().detach().numpy(),
            y_batch=y_batch.cpu().detach().numpy(),
            a_batch=attr,
            explain_func=quantus.explain,
        )
        if s_batch is not None:
            kwargs["s_batch"] = s_batch.cpu().detach().numpy()
        scores[method] = float(np.mean(metric(**kwargs)))
    return scores
