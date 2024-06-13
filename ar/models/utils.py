import torch

def kl_std_normal(mean_squared, var, clamp=False):
    if clamp:
        return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)
    else:
        return 0.5 * (var + mean_squared - torch.log(var) - 1.0)


def noisy_logistic_logpdf(x, loc, scale):
    # Log[ cdf(x + .5) - cdf(x - .5) ]
    #   cdf = lambda x: jax.scipy.stats.logistic.cdf(x, loc=loc, scale=scale)
    cdf = lambda x: torch.sigmoid((x - loc) / scale)
    return torch.log(cdf(x + 0.5) - cdf(x - 0.5))


def unsqueeze_right(x, num_dims=1):
    """Unsqueezes the last `num_dims` dimensions of `x`."""
    return x.view(x.shape + (1,) * num_dims)


def softplus_inverse(x):
    """Helper which computes the inverse of `tf.nn.softplus`."""
    import math
    import numpy as np
    return math.log(np.expm1(x))


SOFTPLUS_INV1 = softplus_inverse(1.0)


def softplus_init1(x):
    """Softplus with a shift to bias the output towards 1.0."""
    return torch.nn.functional.softplus(x + SOFTPLUS_INV1)
    