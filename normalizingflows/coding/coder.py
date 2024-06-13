import numpy as np
import torch

from normalizingflows.utils.distributions import discretized_logistic_cdf, \
    mixture_discretized_logistic_cdf
from . import rans

precision = 31


def get_bins(n_bits):
    """
    Return number of bins and bin_width given n_bits, the number of bits per pixel.

    This n_bins is only valid for well-trained model?
    This might need to be increased for under-fitted model since model might output symbols out of defined CDF.
    n_bins provided here are heuristics since model output symbol most likely fall in this range.
    """
    bin_expo = 17
    if bin_expo == 8:
        bin_expo = 12
    n_bins = 2 ** bin_expo
    bin_width = 1 / 2 ** n_bits
    return n_bins, bin_width


def cdf_fn(z, pz, variable_type, distribution_type, inverse_bin_width):
    if variable_type == 'discrete':
        if distribution_type == 'logistic':
            if len(pz) == 2:
                return discretized_logistic_cdf(
                    z, *pz, inverse_bin_width=inverse_bin_width)
            elif len(pz) == 3:
                return mixture_discretized_logistic_cdf(
                    z, *pz, inverse_bin_width=inverse_bin_width)
        elif distribution_type == 'normal':
            pass

    elif variable_type == 'continuous':
        if distribution_type == 'logistic':
            pass
        elif distribution_type == 'normal':
            pass
        elif distribution_type == 'steplogistic':
            pass
    raise ValueError


def CDF_fn(pz, bin_width, n_bins, variable_type, distribution_type):
    # n_bins will need to be increased if model is not well-trained
    # since there will be a lot of out of expected range symbols

    mean = pz[0] if len(pz) == 2 else pz[0][..., (pz[0].size(-1) - 1) // 2]
    MEAN = torch.round(mean / bin_width).long()

    bin_locations = torch.arange(-n_bins // 2, n_bins // 2)[None, None, None, None, :] + MEAN.cpu()[..., None]
    bin_locations = bin_locations.float() * bin_width
    bin_locations = bin_locations.to(device=pz[0].device)

    pz = [param[:, :, :, :, None] for param in pz]
    cdf = cdf_fn(
        bin_locations - bin_width,
        pz,
        variable_type,
        distribution_type,
        1./bin_width).detach().cpu()

    # Compute CDFs, reweigh to give all bins at least
    # 1 / (2^precision) probability.
    # CDF is equal to floor[cdf * (2^precision - n_bins)] + range(n_bins)
    CDFs = (cdf * ((1 << precision) - n_bins)).long() + torch.arange(n_bins, dtype=torch.int64)
    CDFs = CDFs.to(torch.uint32).int()

    return CDFs, MEAN


def encode_sample(
        z, pz, variable_type, distribution_type, bin_width=1./256, n_bins=256, state=None):
    if state is None:
        state = rans.x_init
    else:
        state = rans.unflatten(state)

    CDFs, MEAN = CDF_fn(pz, bin_width, variable_type, distribution_type)

    # z is transformed to Z to match the indices for the CDFs array
    Z = torch.round(z / bin_width).long() + n_bins // 2 - MEAN
    Z = Z.cpu().numpy()

    if not ((np.sum(Z < 0) == 0 and np.sum(Z >= n_bins-1) == 0)):
        print('Z out of allowed range of values, canceling compression')
        return None

    Z, CDFs = Z.reshape(-1), CDFs.reshape(-1, n_bins).copy()
    for symbol, cdf in zip(Z[::-1], CDFs[::-1]):
        statfun = statfun_encode(cdf)
        state = rans.append_symbol(statfun, precision)(state, symbol)

    state = rans.flatten(state)

    return state


def decode_sample(
        state, pz, variable_type, distribution_type, bin_width=1./256, n_bins=256):
    state = rans.unflatten(state)

    device = pz[0].device
    size = pz[0].size()[0:4]

    CDFs, MEAN = CDF_fn(pz, bin_width, variable_type, distribution_type)

    CDFs = CDFs.reshape(-1, n_bins)
    result = np.zeros(len(CDFs), dtype=int)
    for i, cdf in enumerate(CDFs):
        statfun = statfun_decode(cdf)
        state, symbol = rans.pop_symbol(statfun, precision)(state)
        result[i] = symbol

    Z_flat = torch.from_numpy(result).to(device)
    Z = Z_flat.view(size) - n_bins // 2 + MEAN

    z = Z.float() * bin_width

    state = rans.flatten(state)

    return state, z


def statfun_encode(CDF):
    def _statfun_encode(symbol):
        return CDF[symbol], CDF[symbol + 1] - CDF[symbol]
    return _statfun_encode


def statfun_decode(CDF):
    def _statfun_decode(cf):
        # Search such that CDF[s] <= cf < CDF[s]
        s = np.searchsorted(CDF, cf, side='right')
        s = s - 1
        start, freq = statfun_encode(CDF)(s)
        return s, (start, freq)
    return _statfun_decode


def encode(x, symbol):
    return rans.append_symbol(statfun_encode, precision)(x, symbol)


def decode(x):
    return rans.pop_symbol(statfun_decode, precision)(x)
