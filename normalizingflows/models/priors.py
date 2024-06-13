"""
Collection of flow strategies
"""

from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from normalizingflows.models.utils import Base
from normalizingflows.utils.distributions import sample_discretized_logistic, \
    sample_mixture_discretized_logistic, sample_normal, sample_logistic, \
    sample_discretized_normal, sample_mixture_normal
from .networks import NN
from arithmetic_coder.torchac import torchac
from normalizingflows.coding.coder import CDF_fn, get_bins


def sample_prior(px, variable_type, distribution_type, inverse_bin_width):
    if variable_type == 'discrete':
        if distribution_type == 'logistic':
            if len(px) == 2:
                return sample_discretized_logistic(
                    *px, inverse_bin_width=inverse_bin_width)
            elif len(px) == 3:
                return sample_mixture_discretized_logistic(
                    *px, inverse_bin_width=inverse_bin_width)

        elif distribution_type == 'normal':
            return sample_discretized_normal(
                *px, inverse_bin_width=inverse_bin_width)

    elif variable_type == 'continuous':
        if distribution_type == 'logistic':
            return sample_logistic(*px)
        elif distribution_type == 'normal':
            if len(px) == 2:
                return sample_normal(*px)
            elif len(px) == 3:
                return sample_mixture_normal(*px)
        elif distribution_type == 'steplogistic':
            return sample_logistic(*px)

    raise ValueError


class Prior(Base):
    def __init__(self, size, args):
        super().__init__()
        c, h, w = size

        self.inverse_bin_width = 2**args.n_bits
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type
        self.n_mixtures = args.n_mixtures

        self.train_prior = True

        self.register_buffer('zero_mu', torch.zeros(size=(c, h, w)))
        self.register_buffer('zero_logs', torch.zeros(size=(c, h, w)))

        if self.n_mixtures == 1:
            self.mu = Parameter(torch.Tensor(c, h, w))
            self.logs = Parameter(torch.Tensor(c, h, w))
        elif self.n_mixtures > 1:
            self.mu = Parameter(torch.Tensor(c, h, w, self.n_mixtures))
            self.logs = Parameter(torch.Tensor(c, h, w, self.n_mixtures))
            self.pi_logit = Parameter(torch.Tensor(c, h, w, self.n_mixtures))

        self.reset_parameters()

    def reset_parameters(self):
        self.mu.data.zero_()

        if self.n_mixtures > 1:
            self.pi_logit.data.zero_()
            for i in range(self.n_mixtures):
                self.mu.data[..., i] += i - (self.n_mixtures - 1) / 2.

        self.logs.data.zero_()

    def get_pz(self, n):
        self.train_prior = True
        if not self.train_prior:
            assert self.n_mixtures == 1
            mu = self.zero_mu.expand(n, -1, -1, -1)
            logs = self.zero_logs.expand(n, -1, -1, -1)
            return mu, logs

        if self.n_mixtures == 1:
            mu = self.mu.repeat(n, 1, 1, 1)
            logs = self.logs.repeat(n, 1, 1, 1)  # scaling scale
            return mu, logs

        elif self.n_mixtures > 1:
            pi = F.softmax(self.pi_logit, dim=-1)
            mu = self.mu.repeat(n, 1, 1, 1, 1)
            logs = self.logs.repeat(n, 1, 1, 1, 1)
            pi = pi.repeat(n, 1, 1, 1, 1)
            return mu, logs, pi

    def forward(self, z, ldj):
        pz = self.get_pz(z.size(0))

        return pz, z, ldj

    def sample(self, n):
        pz = self.get_pz(n)

        z_sample = sample_prior(pz, self.variable_type, self.distribution_type, self.inverse_bin_width)

        return z_sample

    def decode(self, states, decode_fn):
        pz = self.get_pz(n=len(states))

        states, z = decode_fn(states, pz)
        return states, z

    def decode_torchac(self, fin):
        pz = self.get_pz(1)

        n_bins, bin_width = get_bins(int(np.log2(self.inverse_bin_width)))

        encoded_length = np.frombuffer(fin.read(np.uint32().itemsize), np.uint32, count=1)[0]
        encoded = fin.read(encoded_length)

        cdf, mean = CDF_fn(pz, bin_width, n_bins, self.variable_type, self.distribution_type)
        # Convert to uint16 first to correctly interpret the decoded bits before convert to long for computation
        Z = torchac.decode_int32_normalized_cdf(cdf, encoded).int().to(mean.device)
        z = (Z - n_bins // 2 + mean).float() * bin_width

        return z


class SplitPrior(Base):
    def __init__(self, c_in, factor_out, height, width, args):
        super().__init__()

        self.split_idx = c_in - factor_out
        self.inverse_bin_width = 2**args.n_bits
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type
        self.input_channel = c_in

        self.train_prior = True

        self.register_buffer('zero_mu', torch.zeros(size=[factor_out, height, width]))
        self.register_buffer('zero_logs', torch.zeros(size=[factor_out, height, width]))

        self.nn = NN(
            args=args,
            c_in=c_in - factor_out,
            c_out=factor_out * 2,
            height=height,
            width=width,
            nn_type=args.splitprior_type)

    def get_py(self, z):
        h = self.nn(z)
        mu = h[:, ::2, :, :]
        logs = h[:, 1::2, :, :]

        py = [mu, logs]

        return py

    def split(self, z):
        z1 = z[:, :self.split_idx, :, :]
        y = z[:, self.split_idx:, :, :]
        return z1, y

    def combine(self, z, y):
        result = torch.cat([z, y], dim=1)

        return result

    def forward(self, z, ldj):
        z, y = self.split(z)

        py = self.get_py(z)

        return py, y, z, ldj

    def inverse(self, z, ldj, y):
        # Sample if y is not given.
        if y is None:
            py = self.get_py(z)
            y = sample_prior(py, self.variable_type, self.distribution_type, self.inverse_bin_width)

        z = self.combine(z, y)

        return z, ldj

    def decode(self, z, ldj, states, decode_fn):
        py = self.get_py(z)
        states, y = decode_fn(states, py)
        return self.combine(z, y), ldj, states

    def decode_torchac(self, z, ldj, fin):
        py = self.get_py(z)

        n_bins, bin_width = get_bins(int(np.log2(self.inverse_bin_width)))

        encoded_length = np.frombuffer(fin.read(np.uint32().itemsize), np.uint32, count=1)[0]
        encoded = fin.read(encoded_length)

        cdf, mean = CDF_fn(py, bin_width, n_bins, self.variable_type, self.distribution_type)
        # Convert to uint16 first to correctly interpret the decoded bits before convert to long for computation
        Y = torchac.decode_int32_normalized_cdf(cdf, encoded).int().to(mean.device)
        y = (Y - n_bins // 2 + mean).float() * bin_width

        return self.combine(z, y), ldj

