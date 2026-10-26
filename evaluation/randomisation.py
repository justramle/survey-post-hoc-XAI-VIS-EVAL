import numpy as np
import quantus

def random_logit(num_classes=1000):
    return quantus.RandomLogit(
        num_classes=num_classes,
        similarity_func=quantus.similarity_func.ssim,
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    )
