import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils_pixelcnn import * 
from model import * 
from PIL import Image
import h5py    
import numpy as np 
import time
import sys
import json
sys.path.append('../')
from utils.data_loader import IDFCompressLocalDataset, CustomDataLoader, get_dataset_hf, get_dataset_local
from utils.transform_utils import build_transform_fn
from utils.expm_utils import get_args_as_obj, config_dict_to_str, get_xid
from utils.transform_utils import uint16_to_uint8
from utils.img_utils import pad_img, crop_img
import tqdm

loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)

def rescaling(x):
    return ((x/255 - .5) * 2.)
    
def evaluate_coding(model, ds, args):
    model.eval()
    mc, mh, mw = (2,32,32)
    bpds = []
    with torch.no_grad():
        for idx, (img, name) in enumerate(tqdm.tqdm(ds, desc='encoding')):
            print(name)
            img = img.cuda()
            c, h, w = img.shape
            padded_img, padding_tuple = pad_img(img, (mh, mw))
            cropped_img = crop_img(rescaling(uint16_to_uint8(padded_img)), mh, mw, batchify=True)
            torch.cuda.synchronize()
            output = []
            for b in range(cropped_img.shape[0]//64+1):
                output1 = model(cropped_img[b*64:(b+1)*64,:,:,:]).cpu()
                output.append(output1)
            output = torch.concatenate(output, axis = 0).cuda()
            print('check datashape ',output.shape, cropped_img.shape)
            loss = loss_op(cropped_img[:], output[:])
            deno = img.numel() * np.log(2.)
            avg_test_loss = loss / deno
            print('bpd ',avg_test_loss)
            with open('{}_results.txt'.format(args.dataset), 'a') as file:
                file.write("{},{:.4f}\n".format(name, avg_test_loss))
            bpds.append(avg_test_loss)
    print("Averaged Results are ",torch.mean(torch.tensor(bpds)), ' bpd')
    with open('{}_results.txt'.format(args.dataset), 'a') as file:
        file.write("avg, {:.4f}\n".format(torch.mean(torch.tensor(bpds))))
    return bpds

def main(args):
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('--dataset', type=str, default='hst',
                            help='input dataset name')
    parser.add_argument('--model', type=str, default='hst',
                            help='input dataset name')
    parser.add_argument('-o', '--save_dir', type=str, default='expms',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-p', '--print_every', type=int, default=100,
                        help='how many steps between print statements')
    parser.add_argument('-t', '--save_every', type=int, default=100,
                        help='Every how many steps to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--flip_horizontal', type=float, default=0.,
                            help='flip input horizontally with given probability (default: 0.)')
    parser.add_argument('--random_crop', action='store_true', default=True,
                        help='flip input horizontally with given probability (default: 0.)')
    parser.add_argument('--input_size', type=str, default='32,32',
                        help='size of input in "h, w" (default "32,32")')
    parser.add_argument('--split_bits', action='store_true', default=True,
                        help='split the 16-bits 1 channel image to 8-bits 2 channels image')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=12,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_steps', type=int,
                        default=2000000, help='How many gradient steps to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('-btr', '--n_batch_train', type=int, default=float('inf'),
                        help='max number of batch for each training epoch. default: iterate over all training data')
    
    parser.add_argument('-btrv', '--n_batch_train_val', type=int, default=float('inf'),
                        help='max number of batch for each train evaluation. default: iterate over all training data')
    
    parser.add_argument('-bv', '--n_batch_val', type=int, default=float('inf'),
                        help='max number of batch for validation epoch. default: iterate over all validation data')
    
    parser.add_argument('-bt', '--n_batch_test', type=int, default=float('inf'),
                        help='max number of batch for test epoch. default: iterate over all test data')

    args = parser.parse_args()
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample_batch_size = 25
    obs = (2, 32, 32)
    input_channels = obs[0]
    
    # -------------------------- load dataset model ----------------------------------------
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(args.dataset)    
    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn)
    test_loader = CustomDataLoader(ds_test, args.batch_size, shuffle=False, max_batch=1)
    
    
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
    
    
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.cuda()
    model_name = args.model
    model.load_state_dict(torch.load(model_name))
    
    evaluate_coding(model,ds_test, args)

if __name__ == '__main__':
    main(sys.argv[1:])

