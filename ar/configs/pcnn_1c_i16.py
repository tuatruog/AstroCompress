import ml_collections
from copy import deepcopy


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.seed = 0
    # config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.restore_ckpt = None

    # Data.
    common_data_cfg = dict(data_spec='lco',
                           split_bits_axis=None,
                           )

    train_data = config.train_data = ml_collections.ConfigDict()
    train_data.update(common_data_cfg)
    train_data.shuffle = True
    train_data.random_crop = True
    train_data.patch_size = (32, 32)
    train_data.batch_size = 64

    eval_data = config.eval_data = ml_collections.ConfigDict()
    eval_data.update(common_data_cfg)
    eval_data.shuffle = False
    eval_data.random_crop = False
    eval_data.patch_size = (64, 64)
    eval_data.batch_size = 1    # Batch size of full-res images.
    eval_data.chunk_size = 32  # For chunking up a large batch of patches and avoid OOM.

    # Train and eval.
    config.training = training = ml_collections.ConfigDict()
    training.n_steps = 1000000
    training.log_metrics_every_steps = 500
    training.checkpoint_every_steps = 5000
    training.eval_every_steps = 20000
    training.max_ckpts_to_keep = 1

    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.use_ema = True  # whether to use EMA params for eval
    evaluate.steps_to_run = None  # Evaluate on the whole eval dataset by default
    #   evaluate.enable_sampling = False  # TODO
    #   evaluate.num_samples = 50000  # TODO
    #   evaluate.enable_fid = False # TODO


    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'pixelcnn'
    model.ema_rate = 0.9999
    model.nr_resnet = 5
    model.nr_filters = 160
    model.nr_logistic_mix = 12


    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.weight_decay = 0
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 100
    #   optim.grad_clip_norm = 1.
    optim.reduce_lr_after = 0.75 * training.n_steps

    return config


def get_cfg_str(config):
    """
    Converts the config produced by get_config() into a string that can be used in the runname.
    We usually extract a few key fields from config that can uniquely identify the run among the
    hyperparameter sweep.
    """
    from collections import OrderedDict
    runname_dict = OrderedDict()
    runname_dict['data'] = config.train_data.data_spec
    runname_dict['split_bits'] = config.train_data.split_bits_axis
    runname_dict['model'] = config.model.name
    runname_dict['nr_resnet'] = config.model.nr_resnet
    runname_dict['nr_filters'] = config.model.nr_filters
    runname_dict['nr_mix'] = config.model.nr_logistic_mix

    from ar.common.run_utils import config_dict_to_str
    return config_dict_to_str(runname_dict, skip_falsy=False)


def get_hyper():
    """
    Produce a list of flattened dicts, each containing a hparam configuration overriding the one in
    get_config(), corresponding to one hparam trial/experiment/work unit.
    :return:
    """
    from ar.common import hyper
    data_specs = ['lco', 'keck', 'hst', 'jwst', 'sdss']
    data_specs = hyper.izip(hyper.sweep('train_data.data_spec', data_specs),
                            hyper.sweep('eval_data.data_spec', data_specs))

    # hparam_cfgs = hyper.product(Ts, noise_sched)
    hparam_cfgs = data_specs
    return hparam_cfgs
