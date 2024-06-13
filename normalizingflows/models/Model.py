import numpy as np
import torch
import tqdm

import normalizingflows.models.generative_flows as generative_flows
from arithmetic_coder.torchac import torchac
from utils.img_utils import crop_img, uncrop_img, pad_img, unpad_img
from normalizingflows.coding.coder import encode_sample, decode_sample, CDF_fn, get_bins
from normalizingflows.models.priors import Prior
from normalizingflows.models.utils import Base
from normalizingflows.optimization.loss import compute_loss_array


class Normalize(Base):
    def __init__(self, args):
        super().__init__()
        self.n_bits = args.n_bits
        self.variable_type = args.variable_type
        self.input_size = args.input_size

    def forward(self, x, ldj, reverse=False):
        domain = 2.**self.n_bits

        if self.variable_type == 'discrete':
            # Discrete variables will be measured on intervals sized 1/domain.
            # Hence, there is no need to change the log Jacobian determinant.
            dldj = 0
        elif self.variable_type == 'continuous':
            dldj = -np.log(domain) * np.prod(self.input_size)
        else:
            raise ValueError

        if not reverse:
            x = (x - domain / 2) / domain
            ldj += dldj
        else:
            x = x * domain + domain / 2
            ldj -= dldj

        return x, ldj


class Model(Base):
    """
    The base VAE class containing gated convolutional encoder and decoder
    architecture. Can be used as a base class for VAE's with normalizing normalizingflows.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type

        self.n_channels, self.height, self.width = args.input_size

        self.normalize = Normalize(args)

        self.flow = generative_flows.GenerativeFlow(
            self.n_channels, self.height, self.width, args)

        self.n_bits = args.n_bits

        self.z_size = self.flow.z_size

        self.prior = Prior(self.z_size, args)

    def dequantize(self, x):
        if self.training:
            x = x + torch.rand_like(x)
        else:
            # Required for stability.
            alpha = 1e-3
            x = x + alpha + torch.rand_like(x) * (1 - 2 * alpha)

        return x

    def loss(self, pz, z, pys, ys, ldj):
        batchsize = z.size(0)
        loss, bpd, bpd_per_prior = \
            compute_loss_array(pz, z, pys, ys, ldj, self.args)

        for module in self.modules():
            if hasattr(module, 'auxillary_loss'):
                loss += module.auxillary_loss() / batchsize

        return loss, bpd, bpd_per_prior

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log
         det jacobian is zero for a plain VAE (without normalizingflows), and z_0 = z_k.
        """
        # Decode z to x.

        # assert x.dtype == torch.int32, f'Type must be int32, current type {x.dtype}'

        x = x.float()

        ldj = torch.zeros_like(x[:, 0, 0, 0])
        if self.variable_type == 'continuous':
            x = self.dequantize(x)
        elif self.variable_type == 'discrete':
            pass
        else:
            raise ValueError

        x, ldj = self.normalize(x, ldj)

        z, ldj, pys, ys = self.flow(x, ldj, pys=(), ys=())

        pz, z, ldj = self.prior(z, ldj)

        loss, bpd, bpd_per_prior = self.loss(pz, z, pys, ys, ldj)

        return loss, bpd, bpd_per_prior, pz, z, pys, ys, ldj

    def inverse(self, z, ys):
        ldj = torch.zeros_like(z[:, 0, 0, 0])
        x, ldj, pys, py = \
            self.flow(z, ldj, pys=[], ys=ys, reverse=True)

        x, ldj = self.normalize(x, ldj, reverse=True)

        x_uint8 = torch.clamp(x, min=0, max=255).to(
                torch.uint8)

        return x_uint8

    def sample(self, n):
        z_sample = self.prior.sample(n)

        ldj = torch.zeros_like(z_sample[:, 0, 0, 0])
        x_sample, ldj, pys, py = \
            self.flow(z_sample, ldj, pys=[], ys=[], reverse=True)

        x_sample, ldj = self.normalize(x_sample, ldj, reverse=True)

        x_sample_uint8 = torch.clamp(x_sample, min=0, max=255).to(
                torch.uint8)

        return x_sample_uint8

    def encode(self, x):
        """
        Encode using rAns (original implementation).

        :param x:
        :return:
        """
        batchsize = x.size(0)
        _, _, _, pz, z, pys, ys, _ = self.forward(x)

        pjs = list(pys) + [pz]
        js = list(ys) + [z]

        states = []

        for b in range(batchsize):
            state = None
            for pj, j in zip(pjs, js):
                pj_b = [param[b:b+1] for param in pj]
                j_b = j[b:b+1]

                state = encode_sample(
                    j_b, pj_b, self.variable_type,
                    self.distribution_type, bin_width=1./(2**self.args.n_bits), state=state)
                if state is None:
                    break

            states.append(state)

        return states

    def decode(self, states):
        """
        Decode using rAns (original implementation).

        :param states:
        :return:
        """
        def decode_fn(states, pj):
            states = list(states)
            j = []

            for b in range(len(states)):
                pj_b = [param[b:b+1] for param in pj]

                states[b], j_b = decode_sample(
                    states[b], pj_b, self.variable_type,
                    self.distribution_type, bin_width=1./(2**self.args.n_bits))

                j.append(j_b)

            j = torch.cat(j, dim=0)
            return states, j

        states, z = self.prior.decode(states, decode_fn=decode_fn)

        ldj = torch.zeros_like(z[:, 0, 0, 0])

        x, ldj = self.flow.decode(z, ldj, states, decode_fn=decode_fn)

        x, ldj = self.normalize(x, ldj, reverse=True)
        x = x.to(dtype=torch.int32)

        return x

    def encode_torchac(self, x, fout):
        """
        Encode using torchac.
        https://github.com/fab-jul/L3C-PyTorch?tab=readme-ov-file#the-torchac-module-fast-entropy-coding-in-pytorch

        :param x: input image CHW
        :param fout: output file
        :return: actual number of bytes for output file
        """
        img, padding_tuple = pad_img(x, (self.height, self.width))

        # Pad left right top bottom
        pl, pr, pt, pb = padding_tuple

        assert pl < 2 ** 16, f'Pad left is {pl} is larger than 16 bits uint'
        assert pr < 2 ** 16, f'Pad right is {pr} is larger than 16 bits uint'
        assert pt < 2 ** 16, f'Pad top is {pt} is larger than 16 bits uint'
        assert pb < 2 ** 16, f'Pad bottom is {pb} is larger than 16 bits uint'

        crop_imgs = crop_img(img, self.height, self.width, batchify=False)

        X, Y, c, h, w = crop_imgs.shape

        assert X < 2 ** 16, f'Height of crop patches {X} is larger than 16 bits uint'
        assert Y < 2 ** 16, f'Height of crop patches {Y} is larger than 16 bits uint'

        crop_imgs = crop_imgs.reshape(X * Y, c, h, w)

        batchsize = crop_imgs.size(0)

        n_bins, bin_width = get_bins(self.n_bits)

        # Keep track of the file size
        num_bytes = 0

        if fout is not None:
            # Write padding tuple
            fout.write(np.uint16(pl).tobytes())
            fout.write(np.uint16(pr).tobytes())
            fout.write(np.uint16(pt).tobytes())
            fout.write(np.uint16(pb).tobytes())
            # Write X, Y for crop patches
            fout.write(np.uint16(X).tobytes())
            fout.write(np.uint16(Y).tobytes())

        # Add number of bytes for header
        num_bytes += 12

        for b in tqdm.tqdm(range(batchsize)):
            _, _, _, pz, z, pys, ys, _ = self.forward(crop_imgs[b:b+1])

            pjs = list(pys) + [pz]
            js = list(ys) + [z]

            for pj, j in reversed(list(zip(pjs, js))):
                pj_b = [param for param in pj]
                j_b = j

                cdf, mean = CDF_fn(pj_b, bin_width, n_bins, self.variable_type, self.distribution_type)
                J = (torch.round(j_b / bin_width).long() + n_bins // 2 - mean)

                # Check if J values are out of bound
                if not (torch.sum(J < 0).item() == 0 and torch.sum(J >= n_bins - 1).item() == 0):
                    print('J out of allowed range of values, canceling compression')
                    return None

                # J's are interpreted as int32 in torchac and expected to be < 2 ** 18 to work properly
                encoded = torchac.encode_int32_normalized_cdf(cdf, J.to(torch.int32).cpu())
                len_bytes = len(encoded)

                assert len_bytes < 2 ** 32, f'Length of encoded bytes {len_bytes} is larger than 32 bits uint'

                if fout is not None:
                    fout.write(np.uint32(len_bytes).tobytes())
                    fout.write(encoded)

                num_bytes += 4
                num_bytes += len_bytes

        return num_bytes

    def decode_torchac(self, fin):
        """
        Decode fits image using torchac.
        https://github.com/fab-jul/L3C-PyTorch?tab=readme-ov-file#the-torchac-module-fast-entropy-coding-in-pytorch

        :param fin: input file
        :return: decoded image
        """

        # dtype for decoded image
        dtype = torch.uint16
        if self.n_channels == 2:
            dtype = torch.uint8

        num_bytes_to_read = np.uint16().itemsize

        padding_tuple = tuple(np.frombuffer(fin.read(num_bytes_to_read * 4), np.uint16, count=4))

        X, Y = np.frombuffer(fin.read(num_bytes_to_read * 2), np.uint16, count=2)

        padded_img = torch.empty((X, Y, self.n_channels, self.height, self.width), dtype=dtype)

        for x in range(X):
            for y in range(Y):
                z = self.prior.decode_torchac(fin)

                ldj = torch.zeros_like(z[:, 0, 0, 0])

                im, ldj = self.flow.decode_torchac(z, ldj, fin)

                im, ldj = self.normalize(im, ldj, reverse=True)

                im = im.to(dtype=dtype)
                padded_img[x, y] = im

        original_img = unpad_img(uncrop_img(padded_img), padding_tuple)

        return original_img
