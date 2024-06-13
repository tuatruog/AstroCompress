import sys
import os
import torch
import argparse
import tqdm
import time
import numpy as np

from utils.data_loader import IDFCompressHfDataset, IDFCompressLocalDataset, get_dataset_hf, get_dataset_local
from utils.transform_utils import uint16_to_uint8
from utils.img_utils import pad_img, crop_img, img_diff_uint16


def main(args):
    parser = argparse.ArgumentParser(description='IDF evaluating for 3d astronomical images neural compression')

    parser.add_argument('--snap_dir_si', type=str, default=None,
                        help='snapshot directory for the single image compressing model')

    parser.add_argument('--snap_dir_res', type=str, default=None,
                        help='snapshot directory for the residual (diff) image compressing model')

    parser.add_argument('--epoch_si', type=int, default=1,
                        help='the single image compressor model epoch to use for evaluation')

    parser.add_argument('--epoch_res', type=int, default=1,
                        help='the residual (diff) image compressor model epoch to use for evaluation')

    parser.add_argument('--dataset', type=str, default=None,
                        help='input dataset name. should be using the full fits image dataset')

    parser.add_argument('--no_inference', action='store_true', default=False,
                        help='disable inference. this will omit inference time and statistic in the report')

    parser.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR', default=None,
                        help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. '
                             'Requires torchac to be installed, see README. '
                             'Files that already exist in WRITE_OUT_DIR are overwritten.')

    parser.add_argument('--no_decode', action='store_true', default=False,
                        help='disable decoding. this will omit decoding time in the report')


    args = parser.parse_args(args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    snapcfg_si = torch.load(args.snap_dir_si + 'cfg.config')
    snapcfg_res = torch.load(args.snap_dir_res + 'cfg.config')

    assert list(snapcfg_si.input_size) == list(snapcfg_res.input_size), 'Models does not have the same input size.'

    if args.write_to_files and not os.path.exists(args.write_to_files):
        os.mkdir(args.write_to_files)

    # remote hugging face dataset
    # (_, _, _ds_test), extract_fn = get_dataset_hf(args.dataset)
    # ds_test = IDFCompressHfDataset(_ds_test, extract_fn)

    # local hugging face dataset
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(args.dataset)
    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    final_model_si = torch.load(args.snap_dir_si + f'idf_epoch_{args.epoch_si}.model')['model']
    final_model_res = torch.load(args.snap_dir_res + f'idf_epoch_{args.epoch_res}.model')['model']
    analytic_bpd, code_bpd, error, N, ts = evaluate_coding(final_model_si, final_model_res, ds_test,
                                                           args, snapcfg_si.input_size, snapcfg_si)

    output = f"""
    {'*' * 80}
    3D COMPRESSION EVALUATION RESULT
    {'*' * 80}
    Model snapshot single image: {args.snap_dir_si}
    Model snapshot diff image: {args.snap_dir_res}
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


def evaluate_coding(model_si, model_res, ds, args, input_size, snapcfg_si):
    model_si.eval()

    mc, mh, mw = input_size

    transform = None
    if snapcfg_si.split_bits:
        transform = uint16_to_uint8

    code_bpds = []
    bpds = []

    N, t_encode, t_inference, t_rans, t_decode, error = [0. for _ in range(6)]

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with (torch.no_grad()):
        for idx, (img, name) in enumerate(tqdm.tqdm(ds, desc='encoding')):
            img = img.to(args.device)
            t, h, w = img.shape

            n_dims = np.prod(img.shape)
            # If split bits, there are actually 2 channels per time step image.
            if snapcfg_si.split_bits:
                n_dims *= 2

            # inference
            starter.record()
            bpd = 0
            for t_step in range(t):
                if t_step == 0:
                    img_t = img[t_step].unsqueeze(0)
                    if transform:
                        img_t = transform(img_t)

                    # If first time step, use single image compressor
                    padded_img, padding_tuple = pad_img(img_t, (mh, mw))
                    cropped_img = crop_img(padded_img, mh, mw, batchify=True)
                    _, bpd_t, _, pz, z, pys, ys, _ = model_si(cropped_img)
                else:
                    img_d = img_diff_uint16(img[t_step], img[t_step - 1]).unsqueeze(0)
                    if transform:
                        img_d = transform(img_d)

                    # else use the residual compressor
                    padded_img, padding_tuple = pad_img(img_d, (mh, mw))
                    cropped_img = crop_img(padded_img, mh, mw, batchify=True)
                    _, bpd_t, _, pz, z, pys, ys, _ = model_res(cropped_img)
                bpd += np.sum(bpd_t.cpu().numpy()) * np.prod((mc, mh, mw))

            bpd /= np.prod(n_dims)

            ender.record()
            torch.cuda.synchronize()
            t_inference += starter.elapsed_time(ender)

            # encode
            tic = time.time()

            # encode using torchac. warning: takes very long for multiple frames
            if args.write_to_files:
                fn = f'{name}_compressed_{t}x{h}x{w}.bin'
                outfn = os.path.join(args.write_to_files, fn)

                num_bytes = 0
                with open(outfn, 'wb') as fout:
                    for t_step in range(img.shape[0]):
                        if t_step == 0:
                            n_byte = model_si.encode_torchac(img[t_step].unsqueeze(0), fout)
                            num_bytes += n_byte
                        else:
                            n_byte = model_res.encode_torchac((img[t_step] - img[t_step - 1]).unsqueeze(0), fout)
                            num_bytes += n_byte

                t_encode += (time.time() - tic)
                code_bpds += [num_bytes * 8 / np.prod(img.shape)]

                if not args.no_decode:
                    # decode
                    recon_im = torch.empty_like(img.shape)
                    with open(outfn, 'rb') as fin:
                        for t_step in range(img.shape[0]):
                            if t_step == 0:
                                im = model_si.decode_torchac(fin)
                                recon_im[t_step, ...] = im.squeeze(0)
                            else:
                                im = model_res.decode_torchac(fin)
                                recon_im[t_step, ...] = recon_im[t_step - 1] + im.squeeze(0)

                    recon_error = torch.sum(torch.abs(recon_im.long().cpu() - im.long().cpu())).item()
                    if recon_error > 0:
                        print(f'Warning: Recon error on {outfn}, error: {recon_error}')
                    error += recon_error
                    t_decode += (time.time() - tic)

            N += 1
            bpds.append(bpd)

    analytic_bpd = np.mean(bpds)
    code_bpd = np.mean(code_bpds)

    ts = [t_encode / N, t_inference / N / 1e3, t_decode / N]

    return analytic_bpd, code_bpd, error, N, ts


if __name__ == '__main__':
    main(sys.argv[1:])