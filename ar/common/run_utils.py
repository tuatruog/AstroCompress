# Methods to help manage experiments.
import numpy as np
import os, sys
import json


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


def get_run_info():
    run_info = {}
    run_info['cmdline'] = " ".join(
        sys.argv)  # attempt to reconstruct the original cmdline; not reliable (e.g., loses quotes)
    run_info['most_recent_version'] = get_git_revision_short_hash()

    for env_var in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID"):  # (xid, wid)
        if env_var in os.environ:
            run_info[env_var] = os.environ[env_var]

    import socket
    run_info['host_name'] = socket.gethostname()
    return run_info


def log_run_info(workdir):
    run_info = get_run_info()
    with open(os.path.join(workdir, f"run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)


# Below from rdc/common_utils.py

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


try:
    from project_configs import args_abbr
except:
    args_abbr = {}


def config_dict_to_str(cfg, record_keys='all', skip_falsy=True, prefix=None, args_abbr=args_abbr):
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


def parse_runname(s, parse_numbers=False):
    """
    Given a string, infer key,value pairs that were used to generate the string and return the corresponding dict.
    Assume that the 'key' and 'value' are separated by '='. Care is taken to handle numbers in scientific notations.
    :param s: a string to be parsed.
    :param parse_numbers: if True, will try to convert values into floats (or ints) if possible; this may potentially
    lose information. By default (False), the values are simply strings as they appear in the original string.
    :return: an ordered dict, with key,value appearing in order
    >>> parse_runname('dir-lamb=2-arch=2_4_8/tau=1.0-step=0-kerasckpt')
    OrderedDict([('lamb', '2'), ('arch', '2_4_8')), ('tau', '1.0'), ('step', '0')])
    >>> parse_runname('rd-ms2020-latent_depth=320-hyperprior_depth=192-lmbda=1e-06-epoch=300-dataset=basenji-data_dim=4-bpp=0.000-psnr=19.875.npz')
    OrderedDict([('latent_depth', '320'),
                 ('hyperprior_depth', '192'),
                 ('lmbda', '1e-06'),
                 ('epoch', '300'),
                 ('dataset', 'basenji'),
                 ('data_dim', '4'),
                 ('bpp', '0.000'),
                 ('psnr', '19.875')])
    """
    from collections import OrderedDict
    import re
    # Want to look for key, value pairs, of the form key_str=val_str.
    # In the following regex, key_str and val_str correspond to the first and second capturing groups, separated by '='.
    # The val_str should either correspond to a sequence of integers separated by underscores (like '2_3_12'), or a
    # numeric expression (possibly in scientific notation), or an alphanumeric string; the regex search is lazy and will
    # stop at the first possible match, in this order.
    # The sub-regex for scientific notation is adapted from https://stackoverflow.com/a/4479455
    sequence_delimiter = "_"
    pattern = fr'(\w+)=((\d+{sequence_delimiter})+\d+|(-?\d*\.?\d+(?:e[+-]?\d+)?)+|\w+)'

    def parse_ints(delimited_ints_str):
        ints = tuple(map(int, delimited_ints_str.split(sequence_delimiter)))
        return ints

    res = OrderedDict()
    for match in re.finditer(pattern, s):
        key = match.group(1)
        val = match.group(2)
        if match.group(3) is not None:  # Non-trivial match for a sequence of ints.
            if parse_numbers:
                val = parse_ints(val)
        else:  # Either matched a float-like number, or some string (\w+).
            if parse_numbers:
                try:
                    val = float(val)
                    if val == int(val):  # Parse to int if this can be done losslessly.
                        val = int(val)
                except ValueError:
                    pass
        res[key] = val
    return res


def preprocess_float_dict(d, format_str='.6g', as_str=False):
    # preprocess the floating values in a dict so that json.dump(dict) looks nice
    import numpy as np
    # import tensorflow as tf
    import torch
    res = {}
    for (k, v) in d.items():
        if isinstance(v, (float, np.floating)) or torch.is_tensor(v):
            if as_str:
                res[k] = format(float(v), format_str)
            else:
                res[k] = float(format(float(v), format_str))
        else:  # if not some kind of float, leave it be
            res[k] = v
    return res


def get_time_str():
    import datetime
    try:
        from configs import strftime_format
    except ImportError:
        strftime_format = "%Y_%m_%d~%H_%M_%S"

    time_str = datetime.datetime.now().strftime(strftime_format)
    return time_str


def natural_sort(l):
    # https://stackoverflow.com/a/4836734
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def psnr_to_float_mse(psnr):
    return 10 ** (-psnr / 10)


def float_mse_to_psnr(float_mse):
    return -10 * np.log10(float_mse)


# My custom logging code for logging in JSON lines ("jsonl") format
import json


class MyJSONEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyJSONEncoder, self).default(obj)


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


""" Run with temporary verbosity """


# from VDM code base

def with_verbosity(temporary_verbosity_level, fn):
    from absl import logging
    old_verbosity_level = logging.get_verbosity()
    logging.set_verbosity(temporary_verbosity_level)
    result = fn()
    logging.set_verbosity(old_verbosity_level)
    return result
