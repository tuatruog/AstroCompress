# Convenience code for notebooks.

import json_lines
import os, glob
from collections import OrderedDict
from ar.common.run_utils import parse_runname
import pandas as pd

def load_jsonl(jsonl, to_df=True):
  with open(jsonl, 'r') as f:
    records = list(json_lines.reader(f))
  if to_df:
    return pd.DataFrame(records)
  return records


def plot_jsonl(jsonl, ax=None, x_key='step', y_key='loss', itv=1, dropna=True, **kwargs):
  import matplotlib.pyplot as plt
  df = load_jsonl(jsonl, to_df=True)
  df = df.set_index(x_key)

  if ax is None:
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', None))
  label = kwargs.pop('label', y_key)
  title = kwargs.pop('title', None)
  y_series = df[y_key]
  if dropna:
    y_series = y_series.dropna()
  ax.plot(y_series[::itv], label=label, **kwargs)
  # [ax.set_xlim(xlim) for ax in axs]
  # [ax.set_ylim(ylim) for ax in axs]
  plt.suptitle(title)
  return ax


def plot_float_imgs(xs, figsize, img_shape=None, titles=None, **imshow_kwargs):
  import matplotlib.pyplot as plt
  import numpy as np
  from common.data_lib import unnormalize_image, quantize_image
  fig, axs = plt.subplots(1, len(xs), figsize=figsize)
  for i in range(len(xs)):
    ax = axs[i]
    img = quantize_image(unnormalize_image(xs[i])).numpy()
    if img_shape:
      img = np.reshape(img, img_shape)
    ax.imshow(img, **imshow_kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    if titles:
      ax.set_title(titles[i])
  plt.subplots_adjust(wspace=0.1, hspace=0)
  return fig



# BEGIN: Copied from https://github.com/yiboyang/eot-descent/blob/master/nbu.py
def get_last_record_from_file(path, as_list=True):
  with open(path, 'r') as f:
    records = list(json_lines.reader(f))
  record = records[-1]  # Use the last step results
  runname_cfg = parse_runname(path, parse_numbers=True)
  record.update(runname_cfg)
  if as_list:
    record = [record]
  return record

def get_records_list_from_file(path):
  with open(path, 'r') as f:
    records = list(json_lines.reader(f))
  runname_cfg = parse_runname(path, parse_numbers=True)
  if 'train/record.jsonl' in path:
    runname_cfg['split'] = 'train'
  else:
    runname_cfg['split'] = 'test'
  records = [{**runname_cfg, **m} for m in records]
  return records

# Small helper to aggregate results from json results files
def get_labels_to_results_df(labels_to_results_globs, results_dir='',
        insert_label=True, records_from_path=get_records_list_from_file, verbose=False):
  # labels_to_results_globs: iterable, label -> [glob_str1, glob_str2], like 'baseline': ['rd-lambda=*.json', 'rd-bpp-c=*.json']
  # results_dir: if provided, will prepend this to the glob pattern
  labels_to_results_df = OrderedDict()  # This concatenates all metrics lists (usually over different lambdas)
  for label, patterns in labels_to_results_globs.items():
    results_files = []
    for pattern in patterns:
      pattern = os.path.join(results_dir, pattern)
      # results_files += tf.io.gfile.glob(pattern)
      results_files += glob.glob(pattern)

    if verbose:
      print(label, results_files, '\n')

    combined_metrics_list = []
    for path in results_files:
        try:
            combined_metrics_list += records_from_path(path)
        except:
            print(f"Bad results in {path}")
            continue

    # labels_to_results_metrics[key] = combined_metrics_list
    labels_to_results_df[label] = pd.DataFrame(combined_metrics_list)
    if insert_label:
      labels_to_results_df[label].insert(0, 'method', label)

  return labels_to_results_df


import functools
import numpy as np
def seldf(df, **conds):
  # Select multiple conditions from df.
  # e.g., sub_df = seldf(df, opt='sgd', split='train', step=10)
  # Based on https://stackoverflow.com/questions/13611065/efficient-way-to-apply-multiple-filters-to-pandas-dataframe-or-series
  conditions = []
  for k,v in conds.items():
    conditions.append(df[k]==v)
  conj = functools.reduce(np.logical_and, conditions)
  return df[conj]

def to_dB(x):
  import numpy as np
  x = np.array(x)
  return -10 * np.log10(x)

# END: Copied from https://github.com/yiboyang/eot-descent/blob/master/nbu.py


def config_matplotlib_for_paper():
  import matplotlib
  matplotlib.rcParams['pdf.fonttype'] = 42
  matplotlib.rcParams['ps.fonttype'] = 42  # avoid type 3 fonts; http://phyletica.org/matplotlib-fonts/
  matplotlib.rcParams['text.usetex'] = True  # latex font
  matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')  # for \text, \mathbb, etc.
  matplotlib.rc('text.latex', preamble=r'\usepackage{bm}')  # use \bm for bold Greek letters like mu


def convert_rate(R, R_unit, want_bits):
  """
  Return a copy of R containing rate information in the desired unit.
  """
  import numpy as np
  assert R_unit in ('nps', 'bps', 'nats', 'bits')
  R = np.array(R)
  if want_bits:
    if R_unit in ('nps', 'nats'):
      return R / np.log(2)
  else:  # want nats
    if R_unit in ('bps', 'bits'):
      return R * np.log(2)
  return R


def gaussian_rdf_intercept(lamb, n):
  """
  Compute the y-intercept of the tangent line with slope -lamb to the R(D) of an n-dimensional Guassian, under MSE
  distortion.
  By differentiating R(D) = -0.5 n log(D) w.r.t. D, we know that the slope at D is equal to R'(D) = -n/(2D).
  Setting this to -lamb, i.e., the tangent point (D_0, R_0) should have its x-coordinate D_0 satisfy -n/(2D_0) = -lamb,
  so D_0 = n/(2lamb), and R_0 = -0.5 n log(D_0). Then it's easy to derive the intercept of the line with slope -lamb
  that passes through (D_0, R_0), i.e., y = -lamb(x - D_0) + R_0 with x = 0
  :param lamb: >=0, negative slope of the tangent line; approaches inf as D
  goes to 0, approaches (and becomes) n/2 as D goes to D_max = var = 1 for a
  standard normal.
  :param n: dimension
  :return:
  """
  lamb = np.atleast_1d(lamb)
  result = n / 2 * (1 - np.log(n / 2 / lamb))
  # When lamb < n/2, the above formula is no longer valid (gives nonsensical negative values); the optimal
  # value of the Lagrangian L(\lambda) = \lambda * D + R in this case equals \lambda * D_max, corresponding to
  # a line with slope lambda passing through (D_max, 0).
  no_meaningful_slope = lamb < n / 2
  D_max = 1
  result[no_meaningful_slope] = D_max * lamb

  return result


def diag_gaussian_rdf(variances, num_points=50, distortion='mse'):
  """
  Compute rate-distortion function of a diagonal Gaussian source, under either squared or mean squared distortion.
  :param variances:
  :param num_points:
  :param distortion:
  :return:
  """
  distortion = distortion.lower()
  assert distortion in ('se', 'mse')
  max_var = np.max(variances)
  n = len(variances)
  lambs = np.linspace(0, max_var, num_points)
  # vars_rep = np.stack([variances] * num_lambdas, axis=0)  # each row is the vector of variances
  vars_rep = np.repeat([variances], num_points, axis=0)  # each row is the vector of variances
  lambs_rep = np.repeat([lambs], n, axis=0).T  # each column is a copy of lambs

  D_mat = np.minimum(vars_rep, lambs_rep)  # reverse water filling
  Rs = 0.5 * np.sum(np.log(vars_rep) - np.log(D_mat), axis=-1)
  Ds = np.sum(D_mat, axis=-1)

  if distortion == 'mse':
    Ds /= n
  return (Ds, Rs)
