import numpy as np
from pathlib import Path
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader  # , TensorDataset
from torchvision import transforms as T

import ml_collections

from utils.data_loader import IDFCompressHfDataset, IDFCompressLocalDataset, get_dataset_hf, get_dataset_local
# from utils.transform_utils import build_transform_fn
from utils.transform_utils import transforms, flip_horizontal_uint16, uint16_to_uint8, convert_numpy


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def build_transform_fn(cfg):
    """
    Process args from command and build transform function.

    :param cfg: a ml_collections.ConfigDict object.
    :return:

    Adapted from utils/transform_utils.py.
    """
    # cfg = ml_collections.ConfigDict(cfg)
    transform_fn = []
    if cfg.random_crop:
        # print("what??",args.input_size[-2:])
        transform_fn.append(transforms.RandomCrop(size=cfg.patch_size))
    if cfg.get('flip_horizontal') and 0. < cfg.flip_horizontal <= 1.:
        transform_fn.append(lambda img: flip_horizontal_uint16(img, cfg.flip_horizontal))
    if cfg.get('split_bits_axis') is not None:
        msb_first = cfg.get('split_bits_msb_first', False)
        split_fn = lambda x: uint16_to_uint8(x, axis=cfg.split_bits_axis, msb_first=msb_first)
        transform_fn.append(split_fn)
    return transforms.Compose(transform_fn)


def get_data_iters(config):
    """
    Get a iterables to (train, eval) data, whose next() method returns a batch of data.
    """
    # args = get_args_as_obj(dict(flip_horizontal=False, split_bits=0, random_crop=0))
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(config.train_data.data_spec)

    train_transform = build_transform_fn(config.train_data)
    ds_train = IDFCompressLocalDataset(root, _ds_train, ext_fn, train_transform)
    eval_transform = build_transform_fn(config.eval_data)
    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn, eval_transform)


    train_iter = DataLoader(ds_train, batch_size=config.train_data.batch_size, shuffle=True,
                            pin_memory=True, num_workers=1)

    eval_iter = DataLoader(ds_test, batch_size=config.eval_data.batch_size, shuffle=False,
                            pin_memory=False, num_workers=1)

    return train_iter, eval_iter

