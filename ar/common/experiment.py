# Template code for training/evaluation experiments. Based on vdm/experiment.py and score_sde_pytorch/run_lib.py
# An Experiment object holds the state of model, optimizer, etc. and provides methods for training and evaluation.

import functools
import ml_collections

from absl import logging
# from clu import parameter_overview
# from clu import checkpoint

import os
import shutil
import pprint
from clu import metric_writers, periodic_actions
from ar.common.custom_writers import create_default_writer
import ar.common.data_lib as data_lib
import ar.common.run_utils as run_utils
from ar.project_configs import TRAIN_COLLECTION, EVAL_COLLECTION, CHECKPOINTS_DIR_NAME
import inspect
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from ar.models import model_utils as mutils
from ar.models.ema import ExponentialMovingAverage

# Keep the import below for registering all model definitions.
from ar.models import pixelcnn
from ar.common.checkpoint_utils import CheckpointManager
from ar.common.custom_writers import Metrics


def get_optimizer(config, params):
    """Returns a torch optimizer object based on `config`.
    Copied from score_sde_pytorch/losses.py
    """
    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                                     eps=config.optim.eps,
                                     weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                                      eps=config.optim.eps,
                                      weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`.
    Copied from score_sde_pytorch/losses.py
    """

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    reduce_lr_after=config.optim.get('reduce_lr_after', -1),
                    grad_clip_norm=config.optim.get('grad_clip_norm', -1)):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if reduce_lr_after >= 0 and step >= reduce_lr_after:
            for g in optimizer.param_groups:
                g['lr'] = lr * 0.1
        if grad_clip_norm >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
        optimizer.step()

    return optimize_fn


def restore_checkpoint(ckpt_path, state, device):
    # Based on score_sde_pytorch/utils.py
    # import tensorflow as tf
    # if not tf.io.gfile.exists(ckpt_path):
    #     tf.io.gfile.makedirs(os.path.dirname(ckpt_path))
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path))
        logging.warning(f"No checkpoint found at {ckpt_path}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_path, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


# def save_checkpoint(ckpt_path, state):
#   # Copied from score_sde_pytorch/utils.py
#   saved_state = {
#     'optimizer': state['optimizer'].state_dict(),
#     'model': state['model'].state_dict(),
#     'ema': state['ema'].state_dict(),
#     'step': state['step']
#   }
#   torch.save(saved_state, ckpt_path)

def extract_state_to_save(state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    return saved_state


def get_step_fn(train, optimize_fn=None,
                ema_eval=False):  # , reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.
    Based on score_sde_pytorch/losses.py
    Args:
      train: whether to create a train or eval step.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      ema_eval: whether to use EMA parameters for evaluation.

    Returns:
      A one-step function for training or evaluation.
    """

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            model.train()
            loss, metrics = model.compute_loss(batch, training=train)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())

            # Log the current learning rate.
            metrics['scalars']['lr'] = optimizer.param_groups[0][
                'lr']  # https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
            return loss, metrics
        else:
            with torch.no_grad():
                if ema_eval:
                    ema = state['ema']
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())

                model.eval()
                if hasattr(model, 'evaluate'):
                    metrics = model.evaluate(batch)
                else:
                    loss, metrics = model.compute_loss(batch, training=train)

                if ema_eval:
                    ema.restore(model.parameters())

            return metrics

    return step_fn


class Experiment:
    """Boilerplate for training and evaluating models. Keeps track of config and data etc.
    Based on vdm/experiment.py"""

    def __init__(self, config: ml_collections.ConfigDict):
        self.config = config

        # Set seed before initializing model.
        seed = config.seed
        # tf.random.set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Automatically set device.
        # logging.info("TF physical devices:\n%s", str(tf.config.list_physical_devices()))
        # tf.config.experimental.set_visible_devices([], 'GPU')   # Hide GPUs from TensorFlow.
        # logging.info("JAX physical devices:\n%s", str(jax.devices()))
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        with config.unlocked():
            config.device = str(device)
        logging.info("Using device: %s", config.device)
        logging.info("Run info:\n%s", pprint.pformat(run_utils.get_run_info()))

        # Initialize data iterators.
        logging.info('=== Initializing dataset ===')
        train_iter, eval_iter = data_lib.get_data_iters(config)
        self.train_iter = data_lib.cycle(train_iter)  # Repeat infinitely.
        self.eval_iter = eval_iter

        # Create/initialize model
        logging.info('=== Creating model/optimizer ===')
        self.model = mutils.create_model(config)
        self.model = self.model.to(config.device)
        _ = self.model.compute_loss(next(self.train_iter), training=True)  # Dummy initialize for LazyModules.
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.model.ema_rate)
        self.optimizer = get_optimizer(config, self.model.parameters())
        # self.optimizer = get_optimizer(
        #   config, [{'params': self.model.score_net.parameters()}, {'params': self.model.encdec.parameters()},
        #            {'params': self.model.gamma.parameters(), 'lr': config.optim.lr * 1000}])
        state = dict(optimizer=self.optimizer, model=self.model, ema=self.ema, step=0)
        logging.info("Using torch model:\n%s", repr(self.model))

        # Restore from checkpoint
        restore_ckpt = self.config.get('restore_ckpt')
        if restore_ckpt:
            ckpt_manager = CheckpointManager(restore_ckpt, file_name='state')
            ckpt_path = ckpt_manager.get_latest_checkpoint_to_restore_from()
            if ckpt_path is not None:
                state = restore_checkpoint(ckpt_path, state, device=config.device)  # Modified in-place.
                logging.info(f"Restored from {ckpt_path}")

        # state['optimizer'] = get_optimizer(config, self.model.parameters()) # Manual re-init.
        self.state = state

        # Create train/eval steps.
        logging.info('=== Initializing train/eval step ===')
        optimize_fn = optimization_manager(config)
        self.train_step_fn = get_step_fn(train=True, optimize_fn=optimize_fn)
        self.eval_step_fn = get_step_fn(train=False, ema_eval=config.eval.use_ema)

        # self.rng, train_rng = jax.random.split(self.rng)
        # self.p_train_step = functools.partial(self.train_step, train_rng)
        # # self.p_train_step = functools.partial(jax.lax.scan, self.p_train_step)  # Allows training for multiple substeps.
        # # self.p_train_step = jax.pmap(self.p_train_step, "batch")

        # self.rng, eval_rng = jax.random.split(self.rng)
        # self.p_eval_step = functools.partial(self.eval_step, eval_rng)

        # self.rng, eval_rng, sample_rng = jax.random.split(self.rng, 3)
        # self.p_eval_step = functools.partial(self.eval_step, eval_rng)
        # self.p_eval_step = jax.pmap(self.p_eval_step, "batch")
        # self.p_sample = functools.partial(
        #     self.sample_fn,
        #     dummy_inputs=next(self.eval_iter)["images"][0],
        #     rng=sample_rng,
        # )
        # self.p_sample = utils.dist(
        #     self.p_sample, accumulate='concat', axis_name='batch')

        logging.info('=== Done with Experiment.__init__ ===')

    # @abstractmethod
    # def sample_fn(self, *, dummy_inputs, rng, params) -> chex.Array:
    #   """Generate a batch of samples in [0, 255]. """
    #   ...
    #
    # @abstractmethod
    # def loss_fn(self, params, batch, rng, is_train) -> Tuple[float, Any]:
    #   """Loss function and metrics."""
    #   ...
    def simple_train_eval_loop(self, config, workdir):
        # config = self.config.train_eval_config

        # Create writers for logs.
        train_dir = os.path.join(workdir, TRAIN_COLLECTION)
        # train_writer = metric_writers.create_default_writer(train_dir, collection=TRAIN_COLLECTION)
        train_writer = create_default_writer(train_dir)
        train_writer.write_hparams(config.to_dict())

        eval_dir = os.path.join(workdir, EVAL_COLLECTION)
        eval_writer = create_default_writer(eval_dir)

        # Get train state.
        state = self.state  # This is a dict containing references to model, optimizer, etc.

        # Set up checkpointing; restore if there is a checkpoint.
        checkpoint_dir = os.path.join(train_dir, CHECKPOINTS_DIR_NAME)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        logging.info("Will save checkpoints to %s", checkpoint_dir)
        max_ckpts_to_keep = config.training.get("max_ckpts_to_keep", 1)
        ckpt_manager = CheckpointManager(checkpoint_dir, file_name='state', max_to_keep=max_ckpts_to_keep)
        ckpt_path = ckpt_manager.get_latest_checkpoint_to_restore_from()
        if ckpt_path is not None:
            state = restore_checkpoint(ckpt_path, state, device=config.device)  # Modified in-place.
            logging.info(f"Restored from {ckpt_path}")

        # # Distribute training.
        # state = flax_utils.replicate(state)
        initial_step = int(state['step'])

        hooks = []
        report_progress = periodic_actions.ReportProgress(
            num_train_steps=config.training.n_steps, writer=train_writer, every_secs=60)
        # if jax.process_index() == 0:
        #   hooks += [report_progress]
        #   if config.get('profile'):
        #     hooks += [periodic_actions.Profile(num_profile_steps=5,
        #                                        logdir=train_dir)]
        hooks += [report_progress]

        step = initial_step
        # substeps = config.substeps  # The number of gradient updates per p_train_step call.
        substeps = 1

        with metric_writers.ensure_flushes(train_writer):
            logging.info('=== Start training ===')
            # the step count starts from 1 to num_train_steps
            while step < config.training.n_steps:
                is_last_step = step + substeps >= config.training.n_steps
                # One training step
                # with jax.profiler.StepTraceAnnotation('train', step_num=step):
                #   batch = jax.tree_map(jnp.asarray, next(self.train_iter))
                #   eval_train = (config.eval_every_steps > 0 and (
                #       step + substeps) % config.eval_every_steps == 0) or is_last_step
                #   state, train_metrics = self.p_train_step(state, batch, eval_train)

                batch = next(self.train_iter)
                # batch = batch.to(config.device) # Model will make sure to move data to the right device.
                loss, train_metrics = self.train_step_fn(state, batch)

                # Quick indication that training is happening.
                logging.log_first_n(
                    logging.WARNING, 'Ran training step %d.', 3, step)
                for h in hooks:
                    h(step)

                new_step = int(state['step'])
                assert new_step == step + substeps
                step = new_step
                # By now, `step` \in [0, num_train_steps] is the number of gradient steps already taken.

                if step % config.training.log_metrics_every_steps == 0 or is_last_step:
                    # metrics = flax_utils.unreplicate(_train_metrics['scalars'])
                    train_metrics = Metrics(train_metrics['scalars'], train_metrics.get('images', {}),
                                            train_metrics.get('texts', {}))
                    train_writer.write_scalars(step, train_metrics.scalars_float)

                if step % config.training.checkpoint_every_steps == 0 or is_last_step:
                    with report_progress.timed('checkpoint'):
                        # ckpt.save(flax_utils.unreplicate(state))
                        ckpt_manager.save(extract_state_to_save(state), step)

                if (
                        config.training.eval_every_steps > 0 and step % config.training.eval_every_steps == 0) or is_last_step:
                    logging.info("Evaluating at step %d", step)
                    with report_progress.timed('eval'):
                        metrics_list = []
                        if config.eval.steps_to_run is None:  # Then we will iterate through the full eval_ds (assumed finite)
                            eval_iter = self.eval_iter
                        else:
                            eval_iter = (x for (i, x) in enumerate(self.eval_iter) if i < config.eval.steps_to_run)
                        val_size = 0
                        for batch in eval_iter:
                            # batch = batch.to(config.device) # Model will make sure to move data to the right device.
                            metrics = self.eval_step_fn(state, batch)
                            metrics = Metrics(metrics['scalars'], metrics.get('images', {}), metrics.get('texts', {}))
                            metrics_list.append(metrics)
                            # val_size += len(batch)
                            # Below does hard coding for the astro dataset. TODO: generalize.
                            x, _ = batch
                            val_size += x.shape[0]
                        # metrics_list = jax.tree_map(float, metrics_list)  # Convert jnp.Array type to scalars, to make tf happy.
                        eval_metrics = Metrics.merge_metrics(metrics_list)
                        eval_writer.write_scalars(step, eval_metrics.scalars_float)
                        # eval_writer.write_images(step, eval_metrics.images)
                        logging.info("Ran validation on %d instances.", val_size)

            logging.info('=== Finished training ===')

        train_writer.close()
        eval_writer.close()

        return eval_dir

    def train_and_evaluate(self, experiments_dir: str, runname: str):
        ##################### BEGIN: slurm-based workdir setup and good old bookkeeping #########################
        xid = run_utils.get_xid()
        # Here, each runname is associated with a different work unit (Slurm call this a 'array job task')
        # within the same experiment. We add the work unit id prefix to make it easier to warm start
        # with the matching wid later.
        wid = run_utils.get_wid()
        if wid is None:
            wid_prefix = ''
        else:
            wid_prefix = f'wid={wid}-'
        workdir = os.path.join(experiments_dir, xid, wid_prefix + runname)
        # e.g., 'train_xms/21965/wid=3-mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        # absl logs from this point on will be saved to files in workdir.
        logging.get_absl_handler().use_absl_log_file(program_name="trainer", log_dir=workdir)

        logging.warning('=== Experiment.train_and_evaluate() ===')
        config = self.config
        logging.info("Using workdir:\n%s", workdir)
        logging.info("Input config:\n%s", pprint.pformat(config))

        # Save the config provided.
        with open(os.path.join(workdir, f"config.json"), "w") as f:
            f.write(config.to_json(indent=2))
        if "config_filename" in config:
            shutil.copy2(config["config_filename"], os.path.join(experiments_dir, xid, "config_script.py"))

        # Log more info.
        logging.info("Run info:\n%s", pprint.pformat(run_utils.get_run_info()))
        run_utils.log_run_info(workdir=workdir)
        # Write a copy of models source code.
        model_source_str = inspect.getsource(inspect.getmodule(self.model))
        with open(os.path.join(workdir, f"model.py"), "w") as f:
            f.write(model_source_str)
        ##################### END: slurm-based workdir setup and good old bookkeeping #########################

        return self.simple_train_eval_loop(self.config, workdir)

    def evaluate(self, logdir, checkpoint_dir):
        ...


# Some quick tools below.

import json


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_workdir_config(workdir, load_latest_ckpt=True, verbose=False):
    """

    :param workdir: e.g., 'train_xms/21965/wid=0-mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
    :param model_cls: if None, will use the 'model.py' saved in the workdir. TODO
    :return:
    """
    workdir = Path(workdir)

    # if expm_cls is None:
    #   src_path = workdir / "main.py"
    #   expm_module = imp.load_source("main_exp", str(src_path))
    #   expm_cls = expm_module.Experiment

    cfg_path = workdir / "config.json"

    config = load_json(cfg_path)
    # config.update(update_config)

    if load_latest_ckpt:
        config['restore_ckpt'] = str(workdir / TRAIN_COLLECTION / CHECKPOINTS_DIR_NAME)

    if verbose:
        logging.info("Will restore from %s", config['restore_ckpt'])

    config = ml_collections.ConfigDict(config)
    return config


def load_experiment(workdir, load_latest_ckpt=True, verbose=False):
    """

    :param workdir: e.g., 'train_xms/21965/wid=0-mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
    :param model_cls: if None, will use the 'model.py' saved in the workdir. TODO
    :param update_config: if provided, will override the config saved in config.json.
    :return:
    """
    config = load_workdir_config(workdir, load_latest_ckpt, verbose)
    expm = Experiment(config)
    return expm
