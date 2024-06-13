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
import sys
import os
import torch
import argparse
import tqdm
import time
import numpy as np

from utils.data_loader import IDFCompressHfDataset, IDFCompressLocalDataset, get_dataset_hf, get_dataset_local
from utils.transform_utils import uint16_to_uint8
from utils.img_utils import pad_img, crop_img


loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

def rescaling(x):
    return ((x/255 - .5) * 2.)
def main(args):
    parser = argparse.ArgumentParser(description='IDF evaluating for 3d astronomical images neural compression')

    parser.add_argument('--snap_dir_si', type=str, default=None,
                        help='snapshot directory for the single image compressing model')

    parser.add_argument('--snap_dir_res', type=str, default=None,
                        help='snapshot directory for the residual (diff) image compressing model')

    parser.add_argument('--epoch_si', type=int, default=1,
                        help='the single image compressor model epoch to use for evaluation')

    parser.add_argument('--epoch_res', type=int, default=1,
                        help='the residual (diff) image compressor model epoch to use for evaluation')

    parser.add_argument('--dataset', type=str, default=None,
                        help='input dataset name. should be using the full fits image dataset')

    parser.add_argument('--no_inference', action='store_true', default=False,
                        help='disable inference. this will omit inference time and statistic in the report')

    parser.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR', default=None,
                        help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. '
                             'Requires torchac to be installed, see README. '
                             'Files that already exist in WRITE_OUT_DIR are overwritten.')

    parser.add_argument('--no_decode', action='store_true', default=False,
                        help='disable decoding. this will omit decoding time in the report')
    parser.add_argument('--random_crop', action='store_true', default=True,
                    help='flip input horizontally with given probability (default: 0.)')
    parser.add_argument('--input_size', type=str, default='32,32',
                        help='size of input in "h, w" (default "32,32")')
    parser.add_argument('--split_bits', action='store_true', default=True,
                        help='split the 16-bits 1 channel image to 8-bits 2 channels image')
    parser.add_argument('--flip_horizontal', type=float, default=0.,
                        help='flip input horizontally with given probability (default: 0.)')

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
    
    args = parser.parse_args(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   
    sample_batch_size = 25
    obs = (2, 32, 32)
    input_channels = obs[0]
    
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
    # -------------------------- load dataset model ----------------------------------------
    # ds_transforms = [rescaling]
    # transform = build_transform_fn(args, ds_transforms)
    
    transform_si = None
    # if args.split_bits:
    #     transform_si = uint16_to_uint8


    # local hugging face dataset
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(args.dataset)
    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn, transform_si)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    final_model_si = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    final_model_si = final_model_si.cuda()
    final_model_si.load_state_dict(torch.load(args.snap_dir_si ))

    final_model_res = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    final_model_res = final_model_si.cuda()
    final_model_res.load_state_dict(torch.load(args.snap_dir_res ))
    
    loss_val = evaluate_coding(final_model_si, final_model_res, ds_test,
                                                           args)


    print('BPD ',loss_val)




def evaluate_coding(model_si, model_res, ds, args):
    mc, mh, mw = (2, 32, 32)
    model_si.eval()
    model_res.eval()
    bpds = []
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with (torch.no_grad()):
        for idx, (img, name) in enumerate(tqdm.tqdm(ds, desc='encoding')):
            bpd = 0
            for t_step in range(img.shape[0]):
                if t_step == 0:
                    # If first time step, use single image compressor
                    padded_img, padding_tuple = pad_img(uint16_to_uint8(img[t_step]), (mh, mw))
                    padded_img = rescaling(padded_img)
                    cropped_img = crop_img(padded_img, mh, mw, batchify=True).cuda()
                    output1 = model_si(cropped_img[:2048,:,:,:]).cpu()
                    output2 = model_si(cropped_img[2048:,:,:,:]).cpu()
                    output = torch.concatenate([output1,output2], axis = 0).cuda()
                else:
                    # else use the residual compressor
                    padded_img, padding_tuple = pad_img(uint16_to_uint8(img[t_step]) - uint16_to_uint8(img[t_step - 1]), (mh, mw))
                    padded_img = rescaling(padded_img)
                    cropped_img = crop_img(padded_img, mh, mw, batchify=True).cuda()
                    output1 = model_si(cropped_img[:2048,:,:,:]).cpu()
                    output2 = model_si(cropped_img[2048:,:,:,:]).cpu()
                    output = torch.concatenate([output1,output2], axis = 0).cuda()
                loss_val = loss_op(cropped_img, output)
                deno = np.prod(cropped_img.shape) * np.log(2.)
                bpd += loss_val/deno
                
            bpd /= img.shape[0]
            print('bpd values: ',bpd)
            torch.cuda.synchronize()
            bpds.append(bpd)
    return bpds


if __name__ == '__main__':
    main(sys.argv[1:])