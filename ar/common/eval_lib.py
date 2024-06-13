from pathlib import Path
from absl import logging
import imp
from ar.common.experiment import TRAIN_COLLECTION, CHECKPOINTS_DIR_NAME, load_experiment
from ar.common.run_utils import parse_runname, load_json, dump_json
from utils.data_loader import IDFCompressHfDataset, IDFCompressLocalDataset, get_dataset_hf, get_dataset_local

import re
import os
import pprint


def eval_workdir(workdir, eval_iter, results_dir,
                 skip_existing=True, **model_evaluate_kwargs):
  """
  Load the latest model ckpt from a given workdir, and evaluate on eval_data by calling
  model.evaluate().
  :param workdir:
  :param eval_data:
  :param results_dir:
  :param model_cls:
  :param skip_existing: if True, will skip the evaluation if a results file with the same name
    already exists.
  :return: results_file_path:
  """

  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    logging.info(f"Created {results_dir} since it doesn't exist")

  # e.g., 'train_xms/21965/wid=3-mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
  # See common.train_lib for details on how workdir is generated.
  workdir = Path(workdir)
  runname = workdir.name
  runname = re.sub(r"^wid=\d+-", "", runname)  # 'mshyper-bpp_c=0.15-latent_ch=320-base_ch=192'
  xid = workdir.parent.name  # '21965'

  expm = load_experiment(workdir, skip_data_loading=True)
  model = expm.model

  model_step = int(expm.state['step'])
  results_file_name = f"{runname}-step={model_step:3g}-xid={xid}.json"
  results_file_path = Path(results_dir) / results_file_name
  if os.path.exists(results_file_path) and skip_existing:
    logging.info(f"Skipping existing results file {results_file_path}")
    return results_file_path

  metrics_list = [model.evaluate(batch, **model_evaluate_kwargs) for batch in eval_iter]
  results_metrics_list = [metric['scalars'] for metric in
                          metrics_list]  # Will be a flat list of dicts

  # Extract hyparameters from runname and add to each dict in results_metrics_list. Useful for
  # parsing results later.
  runname_hparams = parse_runname(runname, parse_numbers=True)
  for instance_id, metrics_dict in enumerate(results_metrics_list):
    metrics_dict['instance_id'] = instance_id
    metrics_dict.update(runname_hparams)

  dump_json(results_metrics_list, results_file_path)
  logging.info(f'Saved results to {results_file_path}')

  return results_file_path
