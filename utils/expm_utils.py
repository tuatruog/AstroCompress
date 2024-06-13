# Useful tools for managing/running experiments.
# From https://github.com/mandt-lab/shallow-ntc/blob/master/common/utils.py and
# https://github.com/mandt-lab/shallow-ntc/blob/3ef7946838558e2bf4a66b228970d206636e784a/common/common_utils.py#L23
# Yibo Yang, 2024

import os, sys, json

def get_xid():
    # See https://slurm.schedmd.com/job_array.html#env_vars
    xid = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    if xid:
        return xid
    xid = os.environ.get("SLURM_JOB_ID", None)
    if xid:
        return xid
    return get_time_str()


def get_wid():
    return os.environ.get("SLURM_ARRAY_TASK_ID", None)

def get_time_str():
    import datetime
    try:
        from configs import strftime_format
    except ImportError:
        strftime_format = "%Y_%m_%d~%H_%M_%S"

    time_str = datetime.datetime.now().strftime(strftime_format)
    return time_str



def get_runname(args_dict, record_keys=tuple(), prefix=''):
  """
  Given a dictionary of cmdline arguments, return a string that identifies the training run.
  :param args_dict:
  :param record_keys: a tuple/list of keys that is a subset of keys in args_dict that will be used to form the runname
  :return:
  """
  kv_strs = []  # ['key1=val1', 'key2=val2', ...]

  for key in record_keys:
    val = args_dict[key]
    if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
      val_str = '_'.join(map(str, val))
    else:
      val_str = str(val)
    kv_strs.append('%s=%s' % (key, val_str))

  return '-'.join([prefix] + kv_strs)


class AttrDict(dict):
  # https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


def get_args_as_obj(args):
  """
  Get an object specifying options/hyper-params from a JSON file or a Python dict; simulates the result of argparse.
  No processing is done if the input is of neither type (assumed to already be in an obj format).
  :param args: either a dict-like object with attributes specifying the model, or the path to some args.json file
  containing the args (which will be loaded and converted to a dict).
  :return:
  """
  if isinstance(args, str):
    import json
    with open(args) as f:
      args = json.load(f)
  if isinstance(args, dict):
    args = AttrDict(args)
  return args


def config_dict_to_str(cfg, record_keys='all', skip_falsy=True, prefix=None,
        args_abbr={}):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    This is really pretty much just a copy of the config_dict_to_str from common_utils.py.
    :param cfg: a dict-like object containing string keys and numeric/list/tuple values.
    :param record_keys: an iterable of strings corresponding to the keys to record. Default ('all') is
    to record every (k,v) pair in the given dict.
    :param skip_falsy: whether to skip keys whose values evaluate to falsy (0, None, False, etc.)
    :param use_abbr: whether to use abbreviations for long key name
    :return: a string like 'key1=20_30-key2=3.4' if the input dict is {'key1': [20, 30], 'key2': 3.4}
    """
    kv_strs = []  # ['key1=val1', 'key2=val2', ...]
    if record_keys == 'all':  # Use all keys.
        record_keys = iter(cfg)
    for key in record_keys:
        val = cfg[key]
        if skip_falsy and not val:
            continue

        if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
            val_str = '_'.join(map(str, val))
        else:
            val_str = str(val)

        if args_abbr:
            key = args_abbr.get(key, key)

        kv_strs.append('%s=%s' % (key, val_str))

    if prefix:
        substrs = [prefix] + kv_strs
    else:
        substrs = kv_strs
    return '-'.join(substrs)

