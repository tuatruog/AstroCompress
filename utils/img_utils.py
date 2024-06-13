import torch
import numpy as np
import torch.nn.functional as F


def crop_img(img, h, w, batchify=True):
    """
    Crop a big image into sequences of smaller images.

    :param img: Image tensor with dim (C, H, W)
    :param h: crop img height
    :param w: crop img width
    :param batchify: boolean for batchifying the return tensor
    :return: cropped image with dim (X, Y, C, h, w) where X, Y is the crop location in the real image or batch of image
    of shape (X * Y, C, h, w)
    """

    C, H, W = img.shape

    assert h == w, f"Input h {h} and w {w} must be equal"
    assert H % h == 0 and W % w == 0, f"Invalid input image shape {img.shape} and input h {h} and w {w}"

    patches = img.unfold(1, h, w).unfold(2, h, w)
    patches = patches.permute(1, 2, 0, 3, 4)

    if batchify:
        X, Y, _, _, _ = patches.shape
        patches = patches.reshape(X * Y, C, h, w)

    assert np.prod(patches.shape) == np.prod(img.shape), \
        f"Result patch shape {patches.shape} does not equal to original image shape {img.shape}"

    return patches


def uncrop_img(patches, X=None, Y=None):
    """
    Specify X and Y when it is not in patches.shape (i.e. when crop_img is called with batchify=True)

    :param patches:
    :param X:
    :param Y:
    :return:
    """
    if X is None and Y is None:
        assert len(patches.shape) == 5, f'Input patches must be of shape (X,Y,C,H,W), got {patches.shape}'

    if bool(X) ^ bool(Y):
        raise Exception('Must specify both X and Y or not specify both')

    if X is not None and Y is not None:
        assert len(patches.shape) == 4, f'Input patches must be of shape (B,C,H,W), got {patches.shape}'
        B, C, h, w = patches.shape
        patches = patches.reshape(X, Y, C, h, w)

    X, Y, C, h, w = patches.shape
    return patches.permute(2, 0, 3, 1, 4).reshape(C, X * h, Y * w)


def pad_img(img, patch_size, padding_mode='reflect'):
    """
    Pad a batch of images so their height and width are divisible by div. Will always pad in the
    bottom right corner for easy inverse.
    :param img: a batch of images; 3D tensor [C, H, W]
    :param patch_size: the integer that the padded image dimension (both height and width) should be
    divisible by.
    :param padding_mode: padding mode
    :return: Padded images as a tensor and padding tuple
    """
    C, H, W = img.size()
    h, w = patch_size
    new_h = ((H + h - 1) // h) * h
    new_w = ((W + w - 1) // w) * w

    if new_h == H and new_w == W:
        return img, (0, 0, 0, 0)

    pad_left = 0
    pad_right = new_w - W
    pad_top = 0
    pad_bottom = new_h - H

    padding_tuple = (pad_left, pad_right, pad_top, pad_bottom)

    img_dtype = img.dtype
    img_padded = F.pad(img.to(torch.int32), padding_tuple, mode=padding_mode).to(img_dtype)

    return img_padded, padding_tuple


def unpad_img(img, padding_tuple):
    """
    Remove padding from a batch of images given the padding tuple.

    :param img: a batch of padded images; 3D tensor [C, H, W]
    :param padding_tuple: a tuple of the form (pad_left, pad_right, pad_top, pad_bottom)
    :return: Unpadded images as a tensor.
    """
    assert len(img.shape) == 3

    c, h, w = img.shape

    pad_left, pad_right, pad_top, pad_bottom = padding_tuple

    start_h = pad_top
    end_h = h - pad_bottom
    start_w = pad_left
    end_w = w - pad_right

    # Slice the tensor to remove the padding
    unpadded_img = img[:, start_h:end_h, start_w:end_w]
    return unpadded_img


def img_diff_uint16(img1, img2):
    """
    Perform img1 - img2 on uint16 tensor.
    """
    assert img1.dtype == torch.uint16
    assert img2.dtype == torch.uint16

    return (img1.view(torch.int16) - img2.view(torch.int16)).view(torch.uint16)


if __name__ == "__main__":
    img = torch.arange(1, 145).reshape((1, 12, 12))
    patch = crop_img(img, 6, 6, batchify=False)
    print(img.shape)
    print(img)
    print(patch)
    print(patch.shape)

    padded_img, padding_tuple = pad_img(img, (7, 7))
    print(padded_img.shape)
    print(padded_img)

    unpadded_img = unpad_img(padded_img, padding_tuple)
    print(unpadded_img.shape)
    print(unpadded_img)

    unpatch = uncrop_img(patch)
    print(unpatch)
