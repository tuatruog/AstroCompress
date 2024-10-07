import os
import numpy as np
import jsonlines

from datasets import load_dataset
from config import ConfigConstants
from constant import DatasetConstants
from dataloader.loader import DatasetMetadata
from dataloader.lris_utils import read_lris


CONFIG_CONST = ConfigConstants()
DATASET_CONST = DatasetConstants()


def get_dataset_metadata_hf(dataset, split_ratio=CONFIG_CONST.DATA_SPLIT_RATIO):
    """
    Librarian function for getting datasets from hugging face and extractor function.
    Extractor is None when we want to return the fits as is.
    Warning: This loads all data to RAM, very large dataset.

    :param dataset:
    :param split_ratio:
    :return: DatasetMetadata
    """
    assert dataset in DATASET_CONST.DATASETS, f'Invalid dataset {dataset}. Must be one of {DATASET_CONST.DATASETS}.'

    hf_path = None
    data_profile = CONFIG_CONST.DATA_PROFILE
    extractor_fn = _extract
    filter_fn = None

    if dataset == DATASET_CONST.HST:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_SBI_16_2D

    elif dataset == DATASET_CONST.JWST:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_SBI_16_3D
        extractor_fn = _extract_jwst_2d

    elif dataset == DATASET_CONST.KECK:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_GBI_16_2D

    elif dataset == DATASET_CONST.SDSS:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_GBI_16_4D
        extractor_fn = _extract_sdss_2d

    elif dataset == DATASET_CONST.JWST_RES:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_SBI_16_3D
        extractor_fn = _extract_jwst_res

    elif dataset == DATASET_CONST.SDSS_RES:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_GBI_16_4D
        extractor_fn = _extract_sdss_res

    elif dataset == DATASET_CONST.JWST_FULL:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_SBI_16_3D
        data_profile = CONFIG_CONST.DATA_PROFILE_FULL
        extractor_fn = None

    elif dataset == DATASET_CONST.SDSS_FULL:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_GBI_16_4D
        data_profile = CONFIG_CONST.DATA_PROFILE_FULL
        extractor_fn = None

    elif dataset == DATASET_CONST.LCO:
        hf_path = DATASET_CONST.ASTRO_COMPRESS_GBI_16_2D_LEGACY
        filter_fn = lambda data: data['telescope'] == 'LCO'

    else:
        raise Exception(f'Unknown dataset {dataset}.')

    return DatasetMetadata(dataset, *_get_dataset(hf_path, data_profile, split_ratio, filter_fn), extractor_fn)


def get_dataset_metadata_local(dataset, split_ratio=CONFIG_CONST.DATA_SPLIT_RATIO):
    """
    Librarian function for getting datasets from cloned repos and extractor function.

    :param dataset:
    :param split_ratio: (1 - split_ratio) is the ratio of validation set.
    :return:
    """
    assert dataset in DATASET_CONST.DATASET, f'Invalid dataset {dataset}. Must be one of {DATASET_CONST.DATASET}.'

    ds_name = None
    filter_fn = lambda sample: True
    extract_fn = _extract_first
    key = 'image'
    test_limit = float('inf')

    if dataset == DATASET_CONST.HST:
        ds_name = DATASET_CONST.SBI_16_2D
        extract_fn = lambda hdul: list(map(_extract, _extract_hst_hdul(hdul)))

    elif dataset == DATASET_CONST.JWST:
        ds_name = DATASET_CONST.SBI_16_3D
        extract_fn = lambda hdul: _extract_jwst_2d(_extract_jwst_hdul(hdul))

    elif dataset == DATASET_CONST.KECK:
        ds_name = DATASET_CONST.GBI_16_2D
        filter_fn = lambda sample: sample['image_id'].startswith('LR.')
        extract_fn = lambda hdul: _extract(_extract_keck_lris_hdul(hdul))

    elif dataset == DATASET_CONST.SDSS:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_2d(_extract_first(hdul))

    elif dataset == DATASET_CONST.SDSS_10:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_2d(_extract_first(hdul))
        test_limit = 10

    elif dataset == DATASET_CONST.HST_5:
        ds_name = DATASET_CONST.SBI_16_2D
        extract_fn = lambda hdul: list(map(_extract, _extract_hst_hdul(hdul)))
        test_limit = 5

    elif dataset == DATASET_CONST.SDSS_10:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_2d(_extract_first(hdul))
        test_limit = 10

    elif dataset == DATASET_CONST.JWST_RES:
        ds_name = DATASET_CONST.SBI_16_3D
        extract_fn = lambda hdul: _extract_jwst_res(_extract_jwst_hdul(hdul))

    elif dataset == DATASET_CONST.JWST_RES1:
        ds_name = DATASET_CONST.SBI_16_3D
        extract_fn = lambda hdul: _extract_jwst_res1(_extract_jwst_hdul(hdul))

    elif dataset == DATASET_CONST.SDSS_RES:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_res(_extract_first(hdul))

    elif dataset == DATASET_CONST.JWST_FULL:
        ds_name = DATASET_CONST.SBI_16_3D
        extract_fn = _extract_jwst_hdul

    elif dataset == DATASET_CONST.SDSS_FULL:
        ds_name = DATASET_CONST.GBI_16_4D

    elif dataset == DATASET_CONST.SDSS_3D:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_3d(_extract_first(hdul))

    elif dataset == DATASET_CONST.SDSS_3T:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_3t(_extract_first(hdul))

    elif dataset == DATASET_CONST.SDSS_4D:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_4d(_extract_first(hdul))

    elif dataset == DATASET_CONST.SDSS_4D_RES:
        ds_name = DATASET_CONST.GBI_16_4D
        extract_fn = lambda hdul: _extract_sdss_4d_res(_extract_first(hdul))

    elif dataset == DATASET_CONST.LCO:
        ds_name = DATASET_CONST.GBI_16_2D_LEGACY
        filter_fn = lambda sample: sample['telescope'] == DATASET_CONST.LCO

    else:
        raise Exception(f'Unknown dataset {dataset}.')

    root = os.path.join(DATASET_CONST.LOCAL_DATA_ROOT, ds_name)
    with jsonlines.open(os.path.join(root, f'splits/{CONFIG_CONST.DATA_PROFILE}_train.jsonl')) as samples:
        train_ids = [sample[key] for sample in samples if filter_fn(sample)]
    with jsonlines.open(os.path.join(root, f'splits/{CONFIG_CONST.DATA_PROFILE}_test.jsonl')) as samples:
        test_ids = [sample[key] for sample in samples if filter_fn(sample)]

    train_ids, val_ids = _split_train_val(train_ids, split_ratio)

    if test_limit < len(test_ids):
        test_ids = test_ids[:test_limit]

    return DatasetMetadata(train_ids, val_ids, test_ids, extract_fn, root)


def _get_dataset(hf_path, name, split_ratio, filter_fn=None):
    """
    Helper function for getting requested dataset.

    :param hf_path:
    :param name:
    :param split_ratio:
    :param filter_fn:
    :return:
    """
    dataset = load_dataset(path=hf_path, name=name, cache_dir=CONFIG_CONST.DATA_CACHE_DIR, trust_remote_code=True)
    ds = dataset.with_format('np')
    ds_train = ds['train']
    ds_test = ds['test']

    if filter_fn:
        ds_train = list(filter(filter_fn, ds_train))
        ds_test = list(filter(filter_fn, ds_test))

    ds_train, ds_val = _split_train_val(ds_train, split_ratio)
    return ds_train, ds_val, ds_test


def _extract(image):
    """
    Extractor that return the same image with channel dimension unsqueezed.

    :param image:
    :return:
    """
    return image[np.newaxis, ...]


def _extract_jwst_2d(image):
    """
    Extractor for getting image of first time step from JWST fits.

    :param image:
    :return:
    """
    return image[0][np.newaxis, ...]


def _extract_sdss_2d(image):
    """
    Extractor for getting image of first time step and the 3rd wavelength channel from SDSS fits.

    :param image:
    :return:
    """
    return image[0, 2][np.newaxis, ...]


def _extract_jwst_res(image):
    """
    Extractor for getting image of residuals between time steps from JWST fits.

    :param image:
    :return:
    """
    assert image.dtype == np.uint16

    def diff(im, t):
        return (im[t].view(np.int16) - im[t - 1].view(np.int16)).view(np.uint16)[np.newaxis, ...]

    res = []
    if image.shape[0] > 1:
        for t_step in range(1, image.shape[0]):
            res.append(diff(image, t_step))

    return res


def _extract_jwst_res1(image):
    """
    Extractor for getting image of residuals between first and second time steps from JWST fits.

    :param image:
    :return:
    """
    assert image.dtype == np.uint16

    return (image[1].view(np.int16) - image[0].view(np.int16)).view(np.uint16)[np.newaxis, ...]


def _extract_sdss_res(image):
    """
    Extractor for getting image of residuals between time steps of 3rd channel from SDSS fits.

    :param image:
    :return:
    """
    assert image.dtype == np.uint16

    def diff(im, t):
        return (im[t, 2].view(np.int16) - im[t - 1, 2].view(np.int16)).view(np.uint16)[np.newaxis, ...]

    res = []
    if image.shape[0] > 1:
        for t_step in range(1, image.shape[0]):
            res.append(diff(image, t_step))

    return res


def _extract_sdss_3d(image):
    """
    Extractor for getting 3d image from the first time step at the middle 3 channels from SDSS fits.

    :param image:
    :return:
    """
    assert image.dtype == np.uint16

    return image[0, 1:4, :, :]


def _extract_sdss_3t(image):
    """
    Extractor for getting 3d image from the first 3 time steps at the 3rd channel from SDSS fits.

    :param image:
    :return:
    """
    assert image.dtype == np.uint16

    if image.shape[0] >= 3:
        return image[0:3, 2, :, :]
    else:
        return None


def _extract_sdss_4d(image):
    """
    Extractor for getting image of multiple channels in the first time step from sdss dataset.

    :param image:
    :return:
    """
    return image[0, 1:4, :, :]


def _extract_sdss_4d_res(image):
    """
    Extractor for getting residual image of multiple channels from sdss dataset.

    :param image:
    :return:
    """
    assert image.dtype == np.uint16

    def diff(im, t):
        return (im[t, 1:4].view(np.int16) - im[t - 1, 1:4].view(np.int16)).view(np.uint16)

    res = []
    if image.shape[0] > 1:
        for t_step in range(1, image.shape[0]):
            res.append(diff(image, t_step))

    return res


def _extract_first(hdul):
    """
    Extract the image from the first header in fits.

    :param hdul:
    :return:
    """
    return hdul[0].data


def _extract_jwst_hdul(hdul):
    """
    Extract fits image from JWST dataset.

    :param hdul:
    :return:
    """
    return hdul['SCI'].data[0]


def _extract_hst_hdul(hdul):
    """
    Extract fits image from HST dataset.

    :param hdul:
    :return:
    """
    return [hdul[1].data, hdul[4].data]


def _extract_keck_lris_hdul(hdul):
    """
    Extract fits image of LRIS KECK dataset.

    :param hdul:
    :return:
    """
    if len(hdul) > 1:
        # multiextension ... paste together the amplifiers
        data, _ = read_lris(hdul)
    else:
        data = hdul[0].data
    return data


def _split_train_val(ds_train, split_ratio):
    """
    Split training dataset to train and validation set based on split_ratio.
    Random seed is set here for split consistency.

    :param ds_train:
    :param split_ratio:
    :return:
    """
    old_state = np.random.get_state()

    rng = np.random.default_rng(seed=42)

    indices = np.arange(len(ds_train))
    rng.shuffle(indices)
    split_idx = int(len(ds_train) * split_ratio)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    ds_train_split = [ds_train[int(i)] for i in train_idx]
    ds_val_split = [ds_train[int(i)] for i in val_idx]

    np.random.set_state(old_state)

    return ds_train_split, ds_val_split
