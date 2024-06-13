import os
import numpy as np
import torch
import tqdm
import jsonlines

from astropy.io import fits
# from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from utils.transform_utils import numpy_to_tensor
from utils.extract_utils import read_lris

from data_configs import DATA_CACHE_DIR, DATA_PROFILE, LOCAL_DATA_ROOT

# 2D dataset name
JWST = 'jwst'
KECK = 'keck'
SDSS = 'sdss'
LCO = 'lco'
HST = 'hst'
HST_5 = 'hst-5'  # hst dataset with only 5 test image
SDSS_10 = 'sdss-10'  # sdss dataset with only 10 test image

# 3D Residual dataset
JWST_RES = 'jwst-res'
JWST_RES1 = 'jwst-res1'
SDSS_RES = 'sdss-res'

# 3D dataset
SDSS_3D = 'sdss-3d'  # image with the 3rd dimension being time dimension
SDSS_3T = 'sdss-3t'

# 4D Residual dataset
SDSS_4D = 'sdss-4d'  # image with multiple channels
SDSS_4D_RES = 'sdss-4d-res'  # residual multi-channel image between different timestep

# Full fits image dataset
JWST_FULL = 'jwst-full'
SDSS_FULL = 'sdss-full'

DATASET = [JWST, KECK, SDSS, LCO, HST, HST_5, SDSS_10, JWST_RES, JWST_RES1, SDSS_RES, JWST_FULL, SDSS_FULL, SDSS_3D, SDSS_3T, SDSS_4D, SDSS_4D_RES]

# Number of channels for each dataset; this is useful for configuring the model input/output channels.
DATASET_NUM_CHANNELS = {
    JWST:1, KECK:1, SDSS:1, LCO:1, HST:1, JWST_RES:1, SDSS_RES:1,
    JWST_FULL:'variable', SDSS_FULL:'variable',     # These vary across different fits files.
    SDSS_3D:3, SDSS_3T:3,
    SDSS_4D:3, SDSS_4D_RES:3    # Need to double check these.
}

class CustomDataLoader(DataLoader):
    """
    A custom torch DataLoader wrapper class that allow user to specify the max number of batches per epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, max_batch=None, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.max_batch = max_batch

    def __iter__(self):
        batch_count = 0
        for batch in super().__iter__():
            if self.max_batch is not None and batch_count >= self.max_batch:
                break
            yield batch
            batch_count += 1


class L3CCustomDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __getitem__(self, idx):
        image = self.ds[idx]

        if self.transform:
            image = self.transform(image)

        return {
            'idx': idx,
            'raw': image,
            'name': idx,
        }

    def __len__(self):
        return len(self.ds)


class AstroCompressHfDataset(Dataset):
    def __init__(self, ds, extract_fn, transform=None):
        self.ds = ds
        self.extract_fn = extract_fn
        self.transform = transform

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image']
        id = item['image_id']

        if self.extract_fn:
            image = self.extract_fn(image)

        image = numpy_to_tensor(image)
        if self.transform:
            image = self.transform(image)

        return image, id

    def __len__(self):
        return len(self.ds)


class IDFCompressHfDataset(AstroCompressHfDataset):
    def __init__(self, ds, extract_fn=None, transform=None):
        super().__init__(ds, extract_fn, transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class L3CCompressHfDataset(AstroCompressHfDataset):
    def __init__(self, ds, extract_fn=None, transform=None):
        super().__init__(ds, extract_fn, transform)

    def __getitem__(self, idx):
        image, id = super().__getitem__(idx)
        return {
            'idx': idx,
            'raw': image,
            'name': id
        }


class AstroCompressLocalDataSet(Dataset):
    def __init__(self, root, image_ids, extract_fn, transform=None):
        """
        Dataset using local cloned astro compress dataset from hugging face.

        :param root: root of data source
        :param image_ids: list of image ids
        :param extract_fn: function to extract image from hdul
        :param transform: transform function
        """

        self.root = root
        self.image_ids = image_ids
        self.extract_fn = extract_fn
        self.transform = transform
        self.residual = False

        self.cache = []
        self.filepath = []

        for image_id in tqdm.tqdm(image_ids, 'Loading fits file for dataset ...'):
            filename = os.path.join(self.root, image_id)
            self.filepath.append(filename)

            with fits.open(filename, memmap=False) as hdul:
                image = extract_fn(hdul)
            if image is not None:
                if isinstance(image, list):
                    for i in range(len(image)):
                        self.cache.append({'id': os.path.basename(image_id)[:-5] + f'.{i}',
                                           'image': numpy_to_tensor(image[i])})
                else:
                    # Copy so that the original image is out of scope, use less RAM
                    self.cache.append({'id': os.path.basename(image_id)[:-5],
                                       'image': numpy_to_tensor(np.copy(image))})

    def __getitem__(self, idx):
        id = self.cache[idx]['id']
        image = self.cache[idx]['image']

        if self.transform:
            image = self.transform(image)

        return image, id

    def __len__(self):
        return len(self.cache)

    def sample_shape(self):
        return self.__getitem__(0)[0].shape


class IDFCompressLocalDataset(AstroCompressLocalDataSet):
    def __init__(self, root, image_ids, extract_fn, transform=None):
        super().__init__(root, image_ids, extract_fn, transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class L3CCompressLocalDataset(AstroCompressLocalDataSet):
    def __init__(self, root, image_ids, extract_fn, transform=None):
        super().__init__(root, image_ids, extract_fn, transform)

    def __getitem__(self, idx):
        image, id = super().__getitem__(idx)
        return {
            'idx': idx,
            'raw': image,
            'name': id
        }


def get_dataset_hf(dataset, split_ratio=0.85):
    """
    Librarian function for getting datasets from hugging face and extractor function.
    Extractor is None when we want to return the fits as is.
    Warning: This loads all data to RAM, very large dataset.

    :param dataset:
    :param split_ratio:
    :return: ds_train, ds_val, ds_test, extractor_fn
    """
    assert dataset in DATASET, f'Invalid dataset {dataset}. Must be one of {DATASET}.'

    if dataset == HST:
        return _get_dataset(hf_path='AstroCompress/SBI-16-2D', name='full', split_ratio=split_ratio), _extract

    elif dataset == JWST:
        return _get_dataset(hf_path='AstroCompress/SBI-16-3D', name=DATA_PROFILE, split_ratio=split_ratio), _extract_jwst_2d

    elif dataset == KECK:
        return _get_dataset(hf_path='AstroCompress/GBI-16-2D', name=DATA_PROFILE, split_ratio=split_ratio), _extract

    elif dataset == SDSS:
        return _get_dataset(hf_path='AstroCompress/GBI-16-4D', name=DATA_PROFILE, split_ratio=split_ratio), _extract_sdss_2d

    elif dataset == JWST_RES:
        return _get_dataset(hf_path='AstroCompress/SBI-16-3D', name=DATA_PROFILE, split_ratio=split_ratio), _extract_jwst_res

    elif dataset == SDSS_RES:
        return _get_dataset(hf_path='AstroCompress/GBI-16-4D', name=DATA_PROFILE, split_ratio=split_ratio), _extract_sdss_res

    elif dataset == JWST_FULL:
        return _get_dataset(hf_path='AstroCompress/SBI-16-3D', name=DATA_PROFILE, split_ratio=split_ratio), None

    elif dataset == SDSS_FULL:
        return _get_dataset(hf_path='AstroCompress/GBI-16-4D', name=DATA_PROFILE, split_ratio=split_ratio), None

    elif dataset == LCO:
        return _get_dataset(hf_path='AstroCompress/GBI-16-2D-Legacy', name=DATA_PROFILE, split_ratio=split_ratio,
                            filter_fn=lambda data: data['telescope'] == 'LCO'), _extract

    raise Exception(f'Unknown dataset {dataset}.')


def _get_dataset(hf_path, name, split_ratio, filter_fn=None):
    """
    Helper function for getting requested dataset.

    :param hf_path:
    :param name:
    :param split_ratio:
    :param filter_fn:
    :return:
    """
    raise Exception('Not working')
    # dataset = load_dataset(path=hf_path, name=name, cache_dir=_CACHE_DIR, trust_remote_code=True)
    # ds = dataset.with_format('np')
    ds = {}
    ds_train = ds['train']
    ds_test = ds['test']

    if filter_fn:
        ds_train = list(filter(filter_fn, ds_train))
        ds_test = list(filter(filter_fn, ds_test))

    ds_train, ds_val = split_train_val(ds_train, split_ratio)
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


def get_dataset_local(dataset, split_ratio=0.85):
    """
    Librarian function for getting datasets from cloned repos and extractor function.

    :param dataset:
    :param split_ratio:
    :return:
    """
    assert dataset in DATASET, f'Invalid dataset {dataset}. Must be one of {DATASET}.'

    filter_fn = lambda sample: True
    extract_fn = _extract_first
    key = 'image'
    test_limit = float('inf')

    if dataset == HST:
        root = os.path.join(LOCAL_DATA_ROOT, 'SBI-16-2D')
        extract_fn = lambda hdul: list(map(_extract, _extract_hst_hdul(hdul)))

    elif dataset == JWST:
        root = os.path.join(LOCAL_DATA_ROOT, 'SBI-16-3D')
        extract_fn = lambda hdul: _extract_jwst_2d(_extract_jwst_hdul(hdul))

    elif dataset == KECK:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-2D')
        filter_fn = lambda sample: sample['image_id'].startswith('LR.')
        extract_fn = lambda hdul: _extract(_extract_keck_lris_hdul(hdul))

    elif dataset == SDSS:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_2d(_extract_first(hdul))

    elif dataset == SDSS_10:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_2d(_extract_first(hdul))
        test_limit = 10

    elif dataset == HST_5:
        root = os.path.join(LOCAL_DATA_ROOT, 'SBI-16-2D')
        extract_fn = lambda hdul: list(map(_extract, _extract_hst_hdul(hdul)))
        test_limit = 5

    elif dataset == SDSS_10:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_2d(_extract_first(hdul))
        test_limit = 10

    elif dataset == JWST_RES:
        root = os.path.join(LOCAL_DATA_ROOT, 'SBI-16-3D')
        extract_fn = lambda hdul: _extract_jwst_res(_extract_jwst_hdul(hdul))

    elif dataset == JWST_RES1:
        root = os.path.join(LOCAL_DATA_ROOT, 'SBI-16-3D')
        extract_fn = lambda hdul: _extract_jwst_res1(_extract_jwst_hdul(hdul))

    elif dataset == SDSS_RES:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_res(_extract_first(hdul))

    elif dataset == JWST_FULL:
        root = os.path.join(LOCAL_DATA_ROOT, 'SBI-16-3D')
        extract_fn = _extract_jwst_hdul

    elif dataset == SDSS_FULL:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')

    elif dataset == SDSS_3D:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_3d(_extract_first(hdul))

    elif dataset == SDSS_3T:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_3t(_extract_first(hdul))

    elif dataset == SDSS_4D:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_4d(_extract_first(hdul))

    elif dataset == SDSS_4D_RES:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-4D')
        extract_fn = lambda hdul: _extract_sdss_4d_res(_extract_first(hdul))

    elif dataset == LCO:
        root = os.path.join(LOCAL_DATA_ROOT, 'GBI-16-2D-Legacy')
        filter_fn = lambda sample: sample['telescope'] == 'LCO'

    else:
        raise Exception(f'Unknown dataset {dataset}.')

    with jsonlines.open(os.path.join(root, f'splits/{DATA_PROFILE}_train.jsonl')) as samples:
        train_ids = [sample[key] for sample in samples if filter_fn(sample)]
    with jsonlines.open(os.path.join(root, f'splits/{DATA_PROFILE}_test.jsonl')) as samples:
        test_ids = [sample[key] for sample in samples if filter_fn(sample)]

    train_ids, val_ids = split_train_val(train_ids, split_ratio)

    if test_limit < len(test_ids):
        test_ids = test_ids[:test_limit]

    return (train_ids, val_ids, test_ids), root, extract_fn


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


def split_train_val(ds_train, split_ratio):
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


if __name__ == '__main__':
    # transform_fn = [transforms.RandomCrop(size=[32, 32]), lambda img: flip_horizontal_uint16(img, 0.5),
    #                 uint16_to_uint8]
    #
    # (_ds_train, _ds_val, _ds_test), ext_fn = get_dataset(JWST)
    # ds_train = IDFCompressDataset(_ds_val, ext_fn, transforms.Compose(transform_fn))
    # print(len(ds_train))
    # print(ds_train[0])
    # print(ds_train[0][0].shape)
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_local(HST_5)

    from utils.transform_utils import uint16_to_uint8

    ds_test = IDFCompressLocalDataset(root, _ds_test, ext_fn)
    print(ds_test[0])
    print(uint16_to_uint8(ds_test[0][0]))
    print(len(ds_test))
    print(ds_test[0][0].shape)
    print(uint16_to_uint8(ds_test[0][0]).shape)

    import psutil

    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
