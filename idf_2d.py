import math
import itertools
import uuid
import os
import json

from datetime import datetime
from utils.data_loader import LCO, KECK, HST, JWST, SDSS
from slurm_template import template

DATASET = 'dataset'
CHANNELS = 'n_channels'
PATCH_SIZE = 'patch_size'
BATCH_SIZE = 'batch_size'
N_FLOWS = 'n_flows'
N_LEVELS = 'n_levels'
N_NET_CHANNELS = 'n_net_channels'
COUPLING_TYPE = 'coupling_type'
DENSENET_DEPTH = 'densenet_depth'
SPLITPRIOR_TYPE = 'splitprior_type'
N_EPOCHS = 'n_epochs'
EVAL_INTERVAL_EPOCH = 'eval_interval_epochs'
EPOCH_LOG_INTERVAL = 'epoch_log_interval'
LR_DECAY_EPOCH = 'lr_decay_epoch'

PARAM_ORDER_DENSE_NET = [DATASET, CHANNELS, PATCH_SIZE, BATCH_SIZE, N_FLOWS, N_LEVELS,
                         COUPLING_TYPE, SPLITPRIOR_TYPE, N_NET_CHANNELS, DENSENET_DEPTH]
PARAM_ORDER_SHALLOW = [DATASET, CHANNELS, PATCH_SIZE, BATCH_SIZE, N_FLOWS, N_LEVELS,
                       COUPLING_TYPE, SPLITPRIOR_TYPE, N_NET_CHANNELS]


dataset_sizes = {LCO: 7, KECK: 109, HST: 1712, JWST: 1018, SDSS: 400}
n_steps = {LCO: 50000, KECK: 200000, HST: 1200000, JWST: 8000000, SDSS: 450000}
DENSENET_TYPE = 'densenet'
SHALLOW_TYPE = 'shallow'


dataset = [LCO, KECK, HST, JWST, SDSS]
channels = [1, 2]
patch_size = [(32, 32), (64, 64)]
batch_size = [32]
n_flows = [4, 8]
n_levels = [2, 3]
n_net_channels = [256, 512]
densenet_depth = [4, 8]


NUM_LOG_PER_VALIDATION = 10
NUM_VALIDATION_PER_TRAIN = 40
NUM_LR_DECAY_PER_TRAIN = 5000


def get_n_epochs(ds, bs):
    return int(n_steps[ds] / math.ceil(dataset_sizes[ds] / bs))


def get_eval_interval_epoch(n_epochs):
    return int(n_epochs / NUM_VALIDATION_PER_TRAIN)


def get_epoch_log_interval(n_epochs):
    return int(n_epochs / NUM_VALIDATION_PER_TRAIN / NUM_LOG_PER_VALIDATION)


def get_lr_decay_epoch(n_epochs):
    return int(n_epochs / NUM_LR_DECAY_PER_TRAIN)


def generate_combinations(*lists):
    return list(itertools.product(*lists))


def map_param(param_order, param_val):
    return {param: val for param, val in zip(param_order, param_val)}


def augment_config(config):
    n_epochs = get_n_epochs(config.get(DATASET), config.get(BATCH_SIZE))
    config[N_EPOCHS] = n_epochs
    config[EVAL_INTERVAL_EPOCH] = get_eval_interval_epoch(n_epochs)
    config[EPOCH_LOG_INTERVAL] = get_epoch_log_interval(n_epochs)
    config[LR_DECAY_EPOCH] = get_lr_decay_epoch(n_epochs)


def get_model_configs():
    combinations_densenet = generate_combinations(dataset, channels, patch_size, batch_size, n_flows, n_levels,
                                                  [DENSENET_TYPE], [DENSENET_TYPE], n_net_channels, densenet_depth)
    combinations_shallow = generate_combinations(dataset, channels, patch_size, batch_size, n_flows, n_levels,
                                                 [SHALLOW_TYPE], [SHALLOW_TYPE], n_net_channels)

    configs = []
    configs.extend([map_param(PARAM_ORDER_DENSE_NET, param_val) for param_val in combinations_densenet])
    configs.extend([map_param(PARAM_ORDER_SHALLOW, param_val) for param_val in combinations_shallow])

    # filter out weird/unusual model combinations to reduce search space
    final_configs = []
    for config in configs:
        if not ((config[N_FLOWS] == 8 and config[N_LEVELS] == 2) or
                (config[N_FLOWS] == 4 and config[N_LEVELS] == 3) or
                (config[N_FLOWS] == 4 and config[N_NET_CHANNELS] == 512) or
                (config[N_FLOWS] == 8 and config[N_NET_CHANNELS] == 256) or
                (DENSENET_DEPTH in config and config[N_LEVELS] == 4 and config[DENSENET_DEPTH] == 8) or
                (config[DATASET] == LCO and config[N_FLOWS] == 4) or
                (config[DATASET] == LCO and config[N_NET_CHANNELS] == 512) or
                (DENSENET_DEPTH in config and config[DATASET] == LCO and config[DENSENET_DEPTH] == 4)):
            augment_config(config)
            final_configs.append(config)
    return final_configs


if __name__ == "__main__":
    configs = get_model_configs()
    print(f'Generated {len(configs)} configurations')

    print('Outputting configurations to files ...')

    base_dir = f'/home/tuannt2/projects/astro-compression/job-config/{datetime.now().strftime("%m_%d_%H_%M")}'
    os.mkdir(base_dir)

    job_mapping = {}

    for config in configs:
        id = str(uuid.uuid4())[:8]
        job_name = id
        srun_command = (f'srun python3 train_idf.py --dataset {config[DATASET]} '
                        f'--outdir /extra/ucibdl1/shared/data/astrocomp/snapshots/idf/{id} '
                        f'--input_size {config[PATCH_SIZE][0]},{config[PATCH_SIZE][1]} '
                        f'{"--split_bits " if config[CHANNELS] == 2 else ""}'
                        f'--random_crop --flip_horizontal 0.5 '
                        f'--batch_size {config[BATCH_SIZE]} '
                        f'--n_flows {config[N_FLOWS]} '
                        f'--n_levels {config[N_LEVELS]} '
                        f'--coupling_type {config[COUPLING_TYPE]} '
                        f'--splitprior_type {config[SPLITPRIOR_TYPE]} '
                        f'--n_channels {config[N_NET_CHANNELS]} '
                        f'--epochs {config[N_EPOCHS]} '
                        f'{f"--densenet_depth {str(config[DENSENET_DEPTH])} " if DENSENET_DEPTH in config else ""}'
                        f'--lr_decay_epoch {config[LR_DECAY_EPOCH]} '
                        f'--evaluate_interval_epochs {config[EVAL_INTERVAL_EPOCH]} '
                        f'--epoch_log_interval {config[EPOCH_LOG_INTERVAL]}')
        job_str = template.format(job_name=id, config_id=f'{id}', srun_command=srun_command)
        job_mapping[id] = srun_command
        with open(base_dir + f'/{id}.job', 'w') as f:
            f.write(job_str)

    with open(base_dir + f'/job_mapping.json', 'w') as f:
        json.dump(job_mapping, f)
