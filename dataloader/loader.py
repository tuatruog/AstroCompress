import os
import numpy as np
import tqdm

from astropy.io import fits
from torch.utils.data import Dataset, DataLoader
from utils.transform_utils import numpy_to_tensor
from dataloader import DATASET_CONST


# Number of channels for each dataset; this is useful for configuring the model input/output channels.
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


class DatasetMetadata:
    TRAIN_SET = 'train'
    VAL_SET = 'val'
    TEST_SET = 'test'

    """
    Container for dataset metadata used for initializing model specific dataset.
    """
    def __init__(self, ds_name, ds_train, ds_val, ds_test, extract_fn, root=None):
        self.ds_name = ds_name
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.extract_fn = extract_fn
        self.root = root


if __name__ == '__main__':
    (_ds_train, _ds_val, _ds_test), root, ext_fn = get_dataset_metadata_hf(DATASET_CONST.HST_5)

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
