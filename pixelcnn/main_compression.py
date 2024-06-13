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
import tqdm



parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('--dataset', type=str, default='hst',
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

model_name = '{}_pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.dataset,args.lr, args.nr_resnet, args.nr_filters)
runname = config_dict_to_str(vars(args),
        record_keys=('dataset', 'nr_resnet', 'nr_filters', 'lr'))
xid = get_xid()
runname = f'xid={xid}-{runname}'
workdir = os.path.join(args.save_dir, runname)
# assert not os.path.exists(workdir)
if not os.path.exists(workdir):
    os.mkdir(workdir)
writer = SummaryWriter(log_dir=os.path.join(workdir, 'logs'))



sample_batch_size = 25
obs = (2, 32, 32)
input_channels = obs[0]
def rescaling(x):
    return ((x/255 - .5) * 2.).astype(np.float32)
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
# -------------------------- load dataset model ----------------------------------------
ds_transforms = [rescaling]
transform = build_transform_fn(args, ds_transforms)
(_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(args.dataset)

ds_train = IDFCompressLocalDataset(root, _ds_train, ext_fn, transform)
ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn, transform)

train_loader = CustomDataLoader(ds_train, args.batch_size, shuffle=True, max_batch=args.n_batch_train)
test_loader = CustomDataLoader(ds_test, args.batch_size, shuffle=True, max_batch=args.n_batch_train)

loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)


model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)


print('starting training')
global_step = 0

while global_step <= args.max_steps:
    model.train(True)
    torch.cuda.synchronize()
    time_ = time.time()
    model.train()
    for batch_idx, (input1, _) in enumerate(train_loader):
        input1 = input1.cuda()
        output = model(input1)
        loss = loss_op(input1, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step % args.print_every == 0 : 
            deno = np.prod(input1.shape) * np.log(2.)
            avg_loss = float(loss) / deno
            writer.add_scalar('train/bpd', avg_loss, global_step)
            # elapsed_time = time.time() - time_
            record = dict(loss=avg_loss, step=global_step)
                    # elapsed_time=elapsed_time)
            print(record)

            with open(os.path.join(workdir, 'train_record.jsonl'), 'a') as f:
                json.dump(record, f)
                f.write('\n')

            time_ = time.time()
        if global_step % args.save_every == 0: 
            # torch.save(model.state_dict(), '.pth'.format(args.dataset, model_name, epoch))
            ckpt_dir = os.path.join(workdir, 'checkpoints')
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            ckpt_save_path = os.path.join(ckpt_dir, f'ckpt-step={global_step}.pth')
            torch.save(model.state_dict(), ckpt_save_path)
            

    # decrease learning rate
    scheduler.step()
    torch.cuda.synchronize()
    if global_step % args.print_every == 0 : 
        model.eval()
        test_loss = 0.

        n_test_instances = 0
        for batch_idx, (input1,_) in enumerate(test_loader):
            input1 = input1.cuda()
            input_var = Variable(input1)
            output = model(input_var)
            loss = loss_op(input_var, output)
            test_loss += loss.item()
            del loss, output
            n_test_instances += len(input1)

        deno = n_test_instances * np.prod(obs) * np.log(2.)
        avg_test_loss = test_loss / deno
        writer.add_scalar('test/bpd', avg_test_loss, global_step)
        print('test loss : %s' % avg_test_loss)

        record = dict(loss=avg_test_loss, step=global_step)
        with open(os.path.join(workdir, 'eval_record.jsonl'), 'a') as f:
            json.dump(record, f)
            f.write('\n')

    

