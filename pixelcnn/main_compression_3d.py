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
sys.path.append('../')
from utils.data_loader import IDFCompressLocalDataset, CustomDataLoader, get_dataset_hf, get_dataset_local
from utils.transform_utils import build_transform_fn

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('--dataset', type=str, default='hst',
                        help='input dataset name')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-p', '--print_every', type=int, default=2,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
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
parser.add_argument('-x', '--max_epochs', type=int,
                    default=100000, help='How many epochs to run in total?')
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

model_name = '3d-{}_pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.dataset,args.lr, args.nr_resnet, args.nr_filters)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))



sample_batch_size = 25
obs = (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
# -------------------------- load dataset model ----------------------------------------
transform = build_transform_fn(args)
(_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(args.dataset)

ds_train = IDFCompressLocalDataset(root, _ds_train, ext_fn, transform)
ds_test = IDFCompressLocalDataset(root, _ds_train, ext_fn, transform)

train_loader = CustomDataLoader(ds_train, args.batch_size, shuffle=True, max_batch=args.n_batch_train)
test_loader = CustomDataLoader(ds_test, args.batch_size, shuffle=True, max_batch=args.n_batch_test)

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
writes = 0
with open('3d{}_results.txt'.format(args.dataset), 'w') as file:
    # Write content to the file
    file.write("{} Results".format(args.dataset))
for epoch in range(args.max_epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input1, _) in enumerate(train_loader):
        zeros = input1[:,0:1,:,:] * 0 
        input1 = torch.concatenate((input1, zeros), axis = 1)
        input1 = input1.cuda()
        output = model(input1)
        loss = loss_op(input1, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (batch_idx +1) % args.print_every == 0 : 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss, {:.4f}'.format(
                (train_loss / deno), 
                (time.time() - time_)))
            with open('3d{}_results.txt'.format(args.dataset), 'a') as file:
                # Append content to the file
                file.write("Loss, {:.4f}\n".format(train_loss / deno))
            train_loss = 0.
            writes += 1
            time_ = time.time()
            

    # decrease learning rate
    scheduler.step()
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input1,_) in enumerate(test_loader):
        zeros = input1[:,0:1,:,:] * 0 
        input1 = torch.concatenate((input1, zeros), axis = 1)
        input1 = input1.cuda()
        input_var = Variable(input1)
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.item()
        del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))
    with open('3d{}_results.txt'.format(args.dataset), 'a') as file:
        # Append content to the file
        file.write("Test Loss, {:.4f}\n".format(test_loss / deno))
        
    if (epoch + 1) % args.save_interval == 0: 
        torch.save(model.state_dict(), '3dmodels_{}_{}_{}.pth'.format(args.dataset, model_name, epoch))

