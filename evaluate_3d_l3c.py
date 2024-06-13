import torch.backends.cudnn
import argparse
import numpy as np
from operator import itemgetter

from l3c.helpers.aligned_printer import AlignedPrinter
from l3c.test.multiscale_tester import MultiscaleTester
from utils.data_loader import L3CCompressLocalDataset, L3CCustomDataset, get_dataset_local, _extract_jwst_res
from utils.transform_utils import uint16_to_uint8
from collections import namedtuple


torch.backends.cudnn.benchmark = True


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--ms_config_p', default='./l3c/configs',
                   help='Path to a multiscale config. All configs should be under folder ./ms/')

    p.add_argument('--log_dir', help='Directory of experiments. Will create a new folder, LOG_DIR_test, to save test '
                                   'outputs.')

    p.add_argument('--log_date',
                   help='log_date for single image compression model, such as 0104_1345.')

    p.add_argument('log_date_res',
                   help='log_date for residual image compression model, such as 0104_1345.')

    p.add_argument('--dataset', type=str, default=None,
                   help='input dataset name')

    p.add_argument('--split_bits', action='store_true', default=False,
                   help='split the 16-bits 1 channel image to 8-bits 2 channels image')

    p.add_argument('--model_input_size', type=str, default='256,256',
                   help='size of input for the model in "h, w" (default "256,256")')

    p.add_argument('--match_filenames', '-fns', nargs='+', metavar='FILTER',
                   help='If given, remove any images in the folders given by IMAGES that do not match any '
                        'of specified filter.')

    p.add_argument('--max_imgs_per_folder', '-m', type=int, metavar='MAX',
                   help='If given, only use MAX images per folder given in IMAGES. Default: None')
    p.add_argument('--crop', type=int, help='Crop all images to CROP x CROP squares. Default: None')

    p.add_argument('--names', '-n', type=str,
                   help='Comma separated list, if given, must be as long as LOG_DATES. Used for output. If not given, '
                        'will just print LOG_DATES as names.')

    p.add_argument('--overwrite_cache', '-f', action='store_true',
                   help='Ignore cached test outputs, and re-create.')
    p.add_argument('--reset_entire_cache', action='store_true',
                   help='Remove cache.')

    p.add_argument('--restore_itr', '-i', default='-1',
                   help='Which iteration to restore for single image model. -1 means latest iteration.'
                        ' Will use closest smaller if exact iteration is not found. Default: -1')

    p.add_argument('--restore_itr_res', '-i', default='-1',
                   help='Which iteration to restore for residual compression model. -1 means latest iteration. '
                        'Will use closest smaller if exact iteration is not found. Default: -1')

    p.add_argument('--recursive', default='0',
                   help='Either an number or "auto". If given, the rgb configs with num_scales == 1 will '
                        'automatically be evaluated recursively (i.e., the RGB baseline). See _parse_recursive_flag '
                        'in multiscale_tester.py. Default: 0')

    p.add_argument('--sample', type=str, metavar='SAMPLE_OUT_DIR',
                   help='Sample from model. Store results in SAMPLE_OUT_DIR.')

    p.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR',
                   help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. If given, the cache is '
                        'ignored and no test output is printed. Requires torchac to be installed, see README. Files '
                        'that already exist in WRITE_OUT_DIR are overwritten.')

    p.add_argument('--compare_theory', action='store_true',
                   help='If given with --write_to_files, will compare actual bitrate on disk to theoretical bitrate '
                        'given by cross entropy.')

    p.add_argument('--time_report', type=str, metavar='TIME_REPORT_PATH',
                   help='If given with --write_to_files, write a report of time needed for each component to '
                        'TIME_REPORT_PATH.')

    p.add_argument('--sort_output', '-s', choices=['testset', 'exp', 'itr', 'res'], default='testset',
                   help='How to sort the final summary. Possible values: "testset" to sort by '
                        'name of the testset // "exp" to sort by experiment log_date // "itr" to sort by iteration // '
                        '"res" to sort by result, i.e., show smaller first. Default: testset')

    flags = p.parse_args()
    flags.cuda = not flags.no_cuda and torch.cuda.is_available()
    flags.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    flags.model_input_size = [int(d) for d in flags.model_input_size.split(',')]

    if flags.compare_theory and not flags.write_to_files:
        raise ValueError('Cannot have --compare_theory without --write_to_files.')
    if flags.write_to_files and flags.sample:
        raise ValueError('Cannot have --write_to_files and --sample.')
    if flags.time_report and not flags.write_to_files:
        raise ValueError('--time_report only valid with --write_to_files.')

    log_date = flags.log_date
    log_date_res = flags.log_date_res

    restore_itr = int(flags.restore_itr)
    restore_itr_res = int(flags.restore_itr_res)

    Testset = namedtuple('Testset', ['id'])
    testset = Testset(id=flags.dataset)

    tester_si = MultiscaleTester(log_date, flags, restore_itr, dataset_type='astro')
    tester_res = MultiscaleTester(log_date, flags, restore_itr, dataset_type='astro')

    transform = None
    if flags.split_bits:
        transform = uint16_to_uint8

    (_, _, _ds_test), root, ext_fn = get_dataset_local(flags.dataset)
    ds_test = L3CCompressLocalDataset(root, _ds_test, ext_fn, transform=transform)

    bpsp = []
    for i, data in enumerate(ds_test):
        img = data['raw']
        first_img = list(img[:1, :, :])
        res_img = _extract_jwst_res(img)
        si_ds = L3CCustomDataset(first_img)
        res_ds = L3CCustomDataset(res_img)

        result_si = tester_si._test(si_ds)
        result_res = tester_res._test(res_ds)

        bpsp.append(np.mean(list(result_si.per_img_results.values()) + list(result_res.per_img_results.values())))

    final_bpsp = np.mean(bpsp)

    print('Testing single image model {} with residual model {} at {} and {} ---'
          .format(log_date, restore_itr, log_date_res, restore_itr_res))

    if not flags.write_to_files:
        print('*** Summary:')
        with AlignedPrinter() as a:
            a.append('Testset', 'Experiment', 'Itr', 'Result')
            a.append(testset.id,  log_date, str(restore_itr), final_bpsp)


if __name__ == '__main__':
    main()
