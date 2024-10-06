import sys
import os
import torch
import argparse
import tqdm
import time
import numpy as np
import json

from utils.data_loader import IDFCompressHfDataset, IDFCompressLocalDataset, get_dataset_hf, get_dataset_local
from utils.transform_utils import uint16_to_uint8
from utils.img_utils import pad_img, crop_img


def main(args):
    parser = argparse.ArgumentParser(description='IDF evaluating for astronomical images neural compression')

    parser.add_argument('--snap_dir', type=str, default=None,
                        help='snapshot directory for the model to evaluate.')

    parser.add_argument('--epoch', type=int, default=1,
                        help='the model epoch to use for evaluation')

    parser.add_argument('--dataset', type=str, default=None,
                        help='input dataset name.')

    parser.add_argument('--no_decode', action='store_true', default=False,
                        help='disable decoding. this will omit decoding time in the report')

    parser.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR', default=None,
                        help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. '
                             'Requires torchac to be installed, see README. '
                             'Files that already exist in WRITE_OUT_DIR are overwritten.')

    args = parser.parse_args(args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    snapcfg = torch.load(args.snap_dir + 'cfg.config')

    if args.write_to_files and not os.path.exists(args.write_to_files):
        os.mkdir(args.write_to_files)

    transform = None
    if snapcfg.split_bits:
        transform = uint16_to_uint8

    # Remote huggingface dataset
    # (_, _, _ds_test), extract_fn = get_dataset_hf(args.dataset)
    # ds_test = IDFCompressHfDataset(_ds_test, extract_fn, transform)

    # Local huggingface dataset
    (_, _, _ds_test), root, ext_fn = get_dataset_local(args.dataset)
    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn, transform)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    final_model = torch.load(args.snap_dir + f'idf_epoch_{args.epoch}.model')['model']
    analytic_bpd, code_bpd, error, N, ts = evaluate_coding(final_model, ds_test, args, snapcfg)

    output = f"""
    {'*' * 80}
    EVALUATION RESULT
    {'*' * 80}
    Model snapshot: {args.snap_dir}
    Test data: {args.dataset}, {N} samples
    Analytical bpd: {analytic_bpd}
    Actual bpd: {code_bpd}
    Reconstruction error: {error}
    Execution time:
    \tTotal time encode:  {ts[0]:.10f} ms
    \tInference:          {ts[1]:.10f} ms
    \tDecode:             {ts[2]:.10f} ms
    {'*' * 80}
    """

    print(output)

    if args.write_to_files:
        with open(os.path.join(args.write_to_files, 'result.txt'), 'w') as f:
            f.write(output)


def evaluate_coding(model, ds, args, snapcfg):
    model.eval()
    if len(snapcfg.input_size) > 2:
        mc, mh, mw = snapcfg.input_size
    else:
        mh, mw = snapcfg.input_size
        if snapcfg.split_bits:
            mc = 2
        else:
            mc = 1

    code_bpds = []
    bpds = []

    report = []

    N, t_encode, t_inference, t_rans, t_decode, error = [0. for _ in range(6)]

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for idx, (img, name) in enumerate(tqdm.tqdm(ds, desc='encoding')):
            img = img.to(args.device)

            c, h, w = img.shape

            # inference
            starter.record()
            padded_img, padding_tuple = pad_img(img, (mh, mw))
            cropped_img = crop_img(padded_img, mh, mw, batchify=True)
            _, bpd, _, pz, z, pys, ys, _ = model(cropped_img)
            ender.record()
            torch.cuda.synchronize()
            t_inf = starter.elapsed_time(ender)
            t_inference += t_inf

            print(t_inf)

            # encode
            tic = time.time()

            if args.write_to_files:
                fn = f'{name}_compressed_{c}x{h}x{w}.bin'
                outfn = os.path.join(args.write_to_files, fn)

                with open(outfn, 'wb') as fout:
                    num_bytes = model.encode_torchac(img, fout)

                t_enc = (time.time() - tic)
                t_encode += t_enc
                code_bpds += [num_bytes * 8 / np.prod(img.shape)]

                print(t_enc)

                if not args.no_decode:
                    # decode
                    with open(outfn, 'rb') as fin:
                        im = model.decode_torchac(fin)
                    recon_error = torch.sum(torch.abs(img.long().cpu() - im.long().cpu())).item()
                    if recon_error > 0:
                        print(f'Warning: Recon error on {outfn}, error: {recon_error}')
                    error += recon_error
                    t_decode += (time.time() - tic)

            N += 1
            bpd_adj = np.sum(bpd.cpu().numpy()) * np.prod((mc, mh, mw)) / np.prod((c, h, w))
            bpds.append(bpd_adj)
            report.append({'image_id': name, 'compression_ratio': snapcfg.n_bits / bpd_adj})

    analytic_bpd = np.mean(bpds)
    code_bpd = np.mean(code_bpds)

    ts = [t_encode / N, t_inference / N, t_decode / N]

    with open(f'./report_{args.dataset}.json', 'w') as jf:
        json.dump(report, jf)

    return analytic_bpd, code_bpd, error, N, ts


if __name__ == '__main__':
    main(sys.argv[1:])
