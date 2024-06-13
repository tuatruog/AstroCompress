from __future__ import print_function

import datetime
import math
import os
import time
import random
import argparse
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

from normalizingflows.models.Model import Model as IDF
from normalizingflows.optimization.training import train, evaluate
from utils.data_loader import IDFCompressHfDataset, IDFCompressLocalDataset, CustomDataLoader, get_dataset_hf, get_dataset_local
from utils.transform_utils import build_transform_fn


def main(args):
    parser = argparse.ArgumentParser(description='IDF training for astronomical images neural compression')

    # ---------------- Operational parameters -------------
    parser.add_argument('--dataset', type=str, default='keck',
                        help='input dataset name')

    parser.add_argument('-od', '--out_dir', type=str, default='./snapshots/idf', metavar='OUT_DIR',
                        help='output directory for model snapshots')

    parser.add_argument('--flip_horizontal', type=float, default=0.,
                        help='flip input horizontally with given probability (default: 0.)')

    parser.add_argument('--random_crop', action='store_true', default=True,
                        help='randomly crop a patch from the original image based on the input size (default: True)')

    parser.add_argument('--input_size', type=str, default='32,32',
                        help='size of input in "h, w" (default "32,32")')

    parser.add_argument('--split_bits', action='store_true', default=False,
                        help='split the 16-bits 1 channel image to 8-bits 2 channels image')

    parser.add_argument('-li', '--log_interval', type=int, default=20, metavar='LOG_INTERVAL',
                        help='number of batches to wait before logging training status')

    parser.add_argument('-eli', '--epoch_log_interval', type=int, default=1, metavar='EPOCH_LOG_INTERVAL',
                        help='number of epoch to wait before logging training status')

    parser.add_argument('--evaluate_interval_epochs', type=int, default=25,
                        help='number of epochs to wait before evaluating the model on validation data')

    parser.add_argument('--manual_seed', type=int,
                        help='manual seed, if not given resorts to random seed.')

    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('-te', '--testing', action='store_false', dest='testing',
                    help='evaluate on test set after training')
    fp.add_argument('-va', '--validation', action='store_true', dest='testing',
                    help='only evaluate on validation set')

    parser.set_defaults(testing=True)
    # ---------------- ----------------------------------- -------------

    # ---------------- Training hyperparameters -------------
    parser.add_argument('-e', '--epochs', type=int, default=2000, metavar='EPOCHS',
                        help='number of epochs to fits (default: 2000)')

    parser.add_argument('-es', '--early_stopping_epochs', type=int, default=300, metavar='EARLY_STOPPING',
                        help='number of early stopping epochs')

    parser.add_argument('-bs', '--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                        help='input batch size for training (default: 256)')

    parser.add_argument('-btr', '--n_batch_train', type=int, default=float('inf'),
                        help='max number of batch for each training epoch. default: iterate over all training data')

    parser.add_argument('-btrv', '--n_batch_train_val', type=int, default=float('inf'),
                        help='max number of batch for each train evaluation. default: iterate over all training data')

    parser.add_argument('-bv', '--n_batch_val', type=int, default=float('inf'),
                        help='max number of batch for validation epoch. default: iterate over all validation data')

    parser.add_argument('-bt', '--n_batch_test', type=int, default=float('inf'),
                        help='max number of batch for test epoch. default: iterate over all test data')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, metavar='LEARNING_RATE',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr_decay', default=0.999, type=float,
                        help='learning rate decay (default: 0.999)')

    parser.add_argument('--lr_decay_epoch', default=1, type=float,
                        help='num epoch per learning rate decay step (default: 1)')

    parser.add_argument('--warmup', type=int, default=10,
                        help='number of warmup epochs (default: 10)')

    parser.add_argument('--data_augmentation_level', type=int, default=2,
                        help='data augmentation level: 1/2 (default: 2)')
    # ---------------- ----------------------------------- -------------

    # ---------------- Normalizing flow model hyperparameters -------------
    parser.add_argument('--variable_type', type=str, default='discrete',
                        help='variable type of data distribution: discrete/continuous (default: discrete)',
                        choices=['discrete', 'continuous'])

    parser.add_argument('--distribution_type', type=str, default='logistic',
                        choices=['logistic', 'normal', 'steplogistic'],
                        help='distribution type for prior: logistic/normal (default: logistic)')

    parser.add_argument('--n_flows', type=int, default=8,
                        help='number of normalizing flows per level for IDF  (default: 8)')

    parser.add_argument('--n_levels', type=int, default=3,
                        help='number of levels for IDF (default: 3)')

    parser.add_argument('--n_channels', type=int, default=512,
                        help='number of channels in coupling and splitprior (factor out) layer (default: 512)')

    parser.add_argument('--coupling_type', type=str, default='densenet',
                        choices=['shallow', 'resnet', 'densenet'],
                        help='type of coupling layer: shallow/resnet/densenet (default: densenet)')

    parser.add_argument('--densenet_depth', type=int, default=12,
                        help='depth of densenets (default: 12)')

    parser.add_argument('--splitfactor', default=0, type=int,
                        help='split factor for coupling layers (default: 0)')

    parser.add_argument('--split_quarter', dest='split_quarter', action='store_true',
                        help='split coupling layer on quarter')
    parser.add_argument('--no_split_quarter', dest='split_quarter', action='store_false')
    parser.set_defaults(split_quarter=True)

    parser.add_argument('--splitprior_type', type=str, default='densenet',
                        choices=['none', 'shallow', 'resnet', 'densenet'],
                        help='type of splitprior. Use \'none\' for no splitprior: shallow/resnet/densenet '
                             '(default: densenet)')

    parser.add_argument('--n_mixtures', type=int, default=5,
                        help='number of mixtures for prior distribution (default: 5)')

    parser.add_argument('--hard_round', dest='hard_round', action='store_true',
                        help='rounding of translation in discrete models. Weird '
                             'probabilistic implications, only for experimental phase')
    parser.add_argument('--no_hard_round', dest='hard_round', action='store_false')
    parser.set_defaults(hard_round=True)

    parser.add_argument('--round_approx', type=str, default='smooth',
                        choices=['smooth', 'stochastic'],
                        help='type of rounding: smooth/stochastic (default: smooth)')

    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature used for BackRound, used in the the SmoothRound module (default=1.0)')
    # ---------------- ----------------------------------- -------------

    args = parser.parse_args(args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.input_size = [int(d) for d in args.input_size.split(',')]

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    args.n_bits = 16 if not args.split_bits else 8

    snap_dir = train_idf(args)

    print(f'Final model saved at {snap_dir}')


def lr_lambda(epoch, warmup, lr_decay):
    return min(1., (epoch + 1) / warmup) * np.power(lr_decay, epoch)


def train_idf(args):
    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = os.path.join(args.out_dir, f'{args.variable_type}_{args.distribution_type}_{args.dataset}')
    snap_dir = snapshots_path

    snap_dir += '_flows_' + str(args.n_flows) + '_levels_' + str(args.n_levels)

    snap_dir = snap_dir + '__' + args.model_signature + '/'

    args.snap_dir = snap_dir

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    transform = build_transform_fn(args)

    # Remote huggingface dataset
    # (_ds_train, _ds_val, _ds_test), extract_fn = get_dataset_hf(args.dataset)
    # ds_train = IDFCompressHfDataset(_ds_train, extract_fn, transform)
    # ds_val = IDFCompressHfDataset(_ds_val, extract_fn, transform)
    # ds_test = IDFCompressHfDataset(_ds_test, extract_fn, transform)

    # Local huggingface dataset
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(args.dataset)
    ds_train = IDFCompressLocalDataset(root, _ds_train, ext_fn, transform)
    ds_val = IDFCompressLocalDataset(root, _ds_val, ext_fn, transform)
    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn, transform)

    train_loader = CustomDataLoader(ds_train, args.batch_size, shuffle=True, max_batch=args.n_batch_train)
    train_val_loader = CustomDataLoader(ds_train, args.batch_size, shuffle=True, max_batch=args.n_batch_train_val)
    val_loader = CustomDataLoader(ds_val, args.batch_size, shuffle=True, max_batch=args.n_batch_val)
    test_loader = CustomDataLoader(ds_test, args.batch_size, shuffle=True, max_batch=args.n_batch_test)

    # update input size to have the correct channel dimension
    args.input_size = ds_train.sample_shape()

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    with open(snap_dir + 'log.txt', 'a') as ff:
        print('\nMODEL SETTINGS: \n', args, '\n', file=ff)

    # SAVING ARGS
    torch.save(args, snap_dir + 'cfg.config')

    # ==================================================================================================================
    # SELECT MODEL
    # ==================================================================================================================
    # flow parameters and architecture choice are passed on to model through args
    idf = IDF(args)
    idf.set_temperature(args.temperature)
    idf.enable_hard_round(args.hard_round)

    model_sample = idf

    # ====================================
    # INIT
    # ====================================
    # data dependend initialization on CPU
    idf.to(args.device)
    for batch_idx, (data, _) in enumerate(train_loader):
        idf(data.to(args.device))

    print("Model initialized")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    idf = torch.nn.DataParallel(idf, dim=0)

    def lr_lambda(epoch):
        return min(1., (epoch + 1) / args.warmup) * np.power(args.lr_decay, epoch)

    optimizer = optim.Adamax(idf.parameters(), lr=args.learning_rate, eps=1.e-7)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    train_bpd = []
    val_bpd = []

    # for early stopping
    best_val_bpd = np.inf
    best_train_bpd = np.inf
    epoch = 0
    best_epoch = 0

    train_times = []

    idf.eval()
    idf.train()

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        tr_loss, tr_bpd = train(epoch, train_loader, idf, optimizer, args)

        if epoch % args.lr_decay_epoch == 0:
            scheduler.step()

        train_bpd.append(tr_bpd)
        train_times.append(time.time() - t_start)

        if ((epoch < args.evaluate_interval_epochs and epoch % (args.evaluate_interval_epochs // 10) == 0)
                or epoch % args.evaluate_interval_epochs == 0):
            v_loss, v_bpd = evaluate(
                train_val_loader, val_loader, idf, model_sample, args,
                epoch=epoch, file=snap_dir + 'log.txt')

            val_bpd.append(v_bpd)

            # Model save based on TRAIN performance (is heavily correlated with validation performance.)
            if np.mean(tr_bpd) < best_train_bpd:
                best_train_bpd = np.mean(tr_bpd)
                best_val_bpd = v_bpd
                best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model': idf.module,
                'optimizer': optimizer,
                'val_bpd': val_bpd
            }, snap_dir + f'idf_epoch_{epoch}.model')
            print('->model saved<-')

            print('(BEST: fits bpd {:.4f}, test bpd {:.4f})'.format(
                best_train_bpd, best_val_bpd))
            print('Average training epoch took %.2f seconds\n' % np.mean(train_times))

            if math.isnan(v_loss):
                raise ValueError('NaN encountered!')

    train_bpd = np.hstack(train_bpd)
    val_bpd = np.array(val_bpd)

    # training time per epoch
    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times)
    print('Average fits time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

    # ==================================================================================================================
    # EVALUATION
    # ==================================================================================================================
    final_model = torch.load(snap_dir + f'idf_epoch_{best_epoch}.model')['model']
    test_loss, test_bpd = evaluate(
        train_val_loader, test_loader, final_model, final_model, args, testing=True,
        epoch=epoch, file=snap_dir + 'test_log.txt')

    print('Test loss / bpd: %.2f / %.2f' % (test_loss, test_bpd))

    return snap_dir


if __name__ == '__main__':
    main(sys.argv[1:])
