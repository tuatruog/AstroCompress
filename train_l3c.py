"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os.path

import torch

# seed at least the random number generators.
# doesn't guarantee full reproducability: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)

# ---

import argparse
import sys
import random

import torch.backends.cudnn
from fjcommon import no_op
import numpy as np

import l3c.pytorch_ext as pe
from l3c.helpers.config_checker import DEFAULT_CONFIG_DIR, ConfigsRepo
from l3c.helpers.global_config import global_config
from l3c.helpers.saver import Saver
from l3c.train.multiscale_trainer import MultiscaleTrainer
from l3c.train.train_restorer import TrainRestorer
from l3c.train.trainer import LogConfig

torch.backends.cudnn.benchmark = True

def _print_debug_info():
    print('*' * 80)
    print(f'DEVICE == {pe.DEVICE} // PyTorch v{torch.__version__}')
    print('*' * 80)


def main(args, configs_dir=DEFAULT_CONFIG_DIR):
    p = argparse.ArgumentParser()

    p.add_argument('--ms_config_p', default='./l3c/configs/ms/cr16b.cf',
                   help='Path to a multiscale config, see README')

    p.add_argument('--dl_config_p', default='./l3c/configs/dl/oi.cf',
                   help='Path to a dataloader config, see README')

    p.add_argument('--log_dir_root', default='./snapshots/l3c',
                   help='All outputs (checkpoints, tensorboard) will be saved here.')

    p.add_argument('--temporary', '-t', action='store_true',
                   help='If given, outputs are actually saved in ${LOG_DIR_ROOT}_TMP.')

    p.add_argument('--log_train', '-ltrain', type=int, default=100,
                   help='Interval of train output.')

    p.add_argument('--log_train_heavy', '-ltrainh', type=int, default=5, metavar='LOG_HEAVY_FAC',
                   help='Every LOG_HEAVY_FAC-th time that i %% LOG_TRAIN is 0, also output heavy logs.')

    p.add_argument('--log_val', '-lval', type=int, default=500,
                   help='Interval of validation output.')

    p.add_argument('-p', action='append', nargs=1,
                   help='Specify global_config parameters, see README')

    p.add_argument('--dataset', type=str, default='keck',
                   help='type of fits data for processing')

    p.add_argument('--input_size', type=str, default='32,32',
                   help='size of input in "h, w" (default "32,32")')

    p.add_argument('--split_bits', action='store_true', default=False,
                   help='split the 16-bits 1 channel image to 8-bits 2 channels image')

    p.add_argument('-bs', '--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                   help='input batch size for training (default: 256)')

    p.add_argument('-btr', '--n_batch_train', type=int, default=float('inf'),
                   help='max number of batch for each training epoch. default: iterate over all training data')

    p.add_argument('-bv', '--n_batch_val', type=int, default=float('inf'),
                   help='max number of batch for validation epoch. default: iterate over all validation data')

    p.add_argument('-bt', '--n_batch_test', type=int, default=float('inf'),
                   help='max number of batch for test epoch. default: iterate over all test data')

    p.add_argument('--load_all', action='store_true', default=False,
                   help='load all data to device (gpu) for faster training (default: False)')

    p.add_argument('--random_crop', action='store_true', default=False,
                   help='randomly crop each image in dataset to specified input size')

    p.add_argument('--flip_horizontal', type=float, default=0.,
                   help='flip input horizontally with given probability (default: 0.)')

    p.add_argument('--restore', type=str, metavar='RESTORE_DIR',
                   help='Path to the log_dir of the model to restore. If a log_date ('
                        'MMDD_HHmm) is given, the model is assumed to be in LOG_DIR_ROOT.')

    p.add_argument('--restore_continue', action='store_true',
                   help='If given, continue in RESTORE_DIR instead of starting in a new folder.')

    p.add_argument('--restore_restart', action='store_true',
                   help='If given, start from iteration 0, instead of the iteration of RESTORE_DIR. '
                        'Means that the model in RESTORE_DIR is used as pretrained model')

    p.add_argument('--restore_itr', '-i', type=int, default=-1,
                   help='Which iteration to restore. -1 means latest iteration. Will use closest smaller if exact '
                        'iteration is not found. Only valid with --restore. Default: -1')

    p.add_argument('--restore_strict', type=str, help='y|n', choices=['y', 'n'], default='y')

    p.add_argument('--num_workers', '-W', type=int, default=8,
                   help='Number of workers used for DataLoader')

    p.add_argument('--manual_seed', type=int,
                   help='manual seed, if not given resorts to random seed.')

    p.add_argument('--saver_keep_tmp_itr', '-si', type=int, default=250)
    p.add_argument('--saver_keep_every', '-sk', type=int, default=10)
    p.add_argument('--saver_keep_tmp_last', '-skt', type=int, default=3)
    p.add_argument('--no_saver', action='store_true',
                   help='If given, no checkpoints are stored.')

    p.add_argument('--debug', action='store_true')

    flags = p.parse_args(args)

    flags.cuda = torch.cuda.is_available()
    flags.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flags.input_size = [int(d) for d in flags.input_size.split(',')]

    if flags.manual_seed is None:
        flags.manual_seed = random.randint(1, 100000)
    random.seed(flags.manual_seed)
    torch.manual_seed(flags.manual_seed)
    np.random.seed(flags.manual_seed)

    _print_debug_info()

    if flags.debug:
        flags.temporary = True

    global_config.add_from_flag(flags.p)
    print(global_config)

    ConfigsRepo(configs_dir).check_configs_available(flags.ms_config_p, flags.dl_config_p)

    saver = (Saver(flags.saver_keep_tmp_itr, flags.saver_keep_every, flags.saver_keep_tmp_last,
                   verbose=True)
             if not flags.no_saver
             else no_op.NoOp())

    restorer = TrainRestorer.from_flags(flags.restore, flags.log_dir_root, flags.restore_continue, flags.restore_itr,
                                        flags.restore_restart, flags.restore_strict)

    trainer = MultiscaleTrainer(flags, flags.ms_config_p, flags.dl_config_p,
                                flags.log_dir_root + ('_TMP' if flags.temporary else ''),
                                LogConfig(flags.log_train, flags.log_val, flags.log_train_heavy),
                                flags.num_workers,
                                saver=saver, restorer=restorer)
    if not flags.debug:
        trainer.train()
    else:
        trainer.debug()


if __name__ == '__main__':
    main(sys.argv[1:])
