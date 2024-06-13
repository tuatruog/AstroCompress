import random
import torch
import numpy as np

from torchvision import transforms


def numpy_to_tensor(arr):
    """
    Converting np.ndarray to torch.Tensor with dtype uint16

    :param arr:
    :return:
    """
    if isinstance(arr, torch.Tensor):
        if arr.dtype == torch.uint16:
            return arr
        return arr.view(dtype=torch.uint16)

    if not isinstance(arr, np.ndarray):
        raise ValueError("Input should be a NumPy ndarray or a tensor.")

    return torch.from_numpy(arr.view(np.uint16))


def uint16_to_uint8(x, axis=0, msb_first=False):
    """
    Convert a uint16 tensor to a uint8 tensor, doubling the specified dimension.
    
    Parameters:
    - x: torch.Tensor of shape [c, h, w] and dtype torch.uint16
    - axis: int, the dimension to double (0 for channel, 1 for height, 2 for width)
    - msb_first: bool, if True, the most significant byte (MSB) comes before the least significant byte (LSB)
    
    Returns:
    - torch.Tensor of shape with the specified dimension doubled and dtype torch.uint8
    """
    if axis not in [0, 1, 2]:
        raise ValueError("axis must be 0, 1, or 2")
    
    # Step 1: Convert to uint8 in the default order
    x_uint8 = x.contiguous().view(torch.uint8)
    
    # Step 2: Reshape the uint8 tensor to separate the bytes
    x_uint8_reshaped = x_uint8.view(*x.shape, 2)
    
    # Step 3: Optionally reverse the order of the bytes
    if msb_first:
        x_uint8_reshaped = x_uint8_reshaped.flip(dims=[-1])
    
    # Step 4: Reshape to double the specified dimension
    if axis == 0:
        # Double the channel dimension
        x_uint8_reordered = x_uint8_reshaped.permute(0, 3, 1, 2).contiguous()
        x_uint8_final = x_uint8_reordered.view(2 * x.shape[0], x.shape[1], x.shape[2])
    elif axis == 1:
        # Double the height dimension
        x_uint8_reordered = x_uint8_reshaped.permute(0, 1, 3, 2).contiguous()
        x_uint8_final = x_uint8_reordered.view(x.shape[0], 2 * x.shape[1], x.shape[2])
    elif axis == 2:
        # Double the width dimension
        x_uint8_reordered = x_uint8_reshaped.permute(0, 1, 2, 3).contiguous()
        x_uint8_final = x_uint8_reordered.view(x.shape[0], x.shape[1], 2 * x.shape[2])
    
    return x_uint8_final


def flip_horizontal_uint16(img, p):
    """
    Horizontal flip for image with dim CHW that work with dtype=torch.uint16.

    :param img:
    :param p:
    :return:
    """
    if random.random() < p:
        return img.view(torch.int16).flip(-1).view(torch.uint16)
    return img

def convert_numpy(x):
    return x.numpy()
    
def build_transform_fn(args, extra_fn = None):
    """
    Process args from command and build transform function.

    :param args:
    :return:
    """
    transform_fn = []
    if args.random_crop:
        transform_fn.append(transforms.RandomCrop(size=args.input_size[-2:]))
    if args.flip_horizontal and 0. < args.flip_horizontal <= 1.:
        transform_fn.append(lambda img: flip_horizontal_uint16(img, args.flip_horizontal))
    if args.split_bits:
        transform_fn.append(uint16_to_uint8)
    if extra_fn:
        transform_fn.append(convert_numpy)
        transform_fn.extend(extra_fn)
    return transforms.Compose(transform_fn)


if __name__ == "__main__":
    # Test uint16_to_uint8 conversion.
    np.random.seed(1)
    torch.manual_seed(1)
    x = torch.randint(0, 2**16-1, [2, 4, 3], dtype=torch.uint16)
    x_32 = x.to(torch.int32)

    def get_outshape(x, axis):
        out_shape = list(x.shape)
        out_shape[axis] *= 2
        return tuple(out_shape)

    axis = 0
    out_shape = get_outshape(x, axis)
    u = uint16_to_uint8(x, axis=axis, msb_first=False).to(torch.int32)
    assert u.shape == out_shape
    assert torch.all(u[0] + u[1] * 2**8 == x_32[0])
    u = uint16_to_uint8(x, axis=axis, msb_first=True).to(torch.int32)
    assert u.shape == out_shape
    assert torch.all(u[1] + u[0] * 2**8 == x_32[0])

    axis = 1
    out_shape = get_outshape(x, axis)
    u = uint16_to_uint8(x, axis=axis, msb_first=False).to(torch.int32)
    assert u.shape == out_shape
    assert torch.all(u[:, 0] + u[:, 1]*2**8 == x_32[:, 0])
    u = uint16_to_uint8(x, axis=axis, msb_first=True).to(torch.int32)
    assert u.shape == out_shape
    assert torch.all(u[:, 1] + u[:, 0]*2**8 == x_32[:, 0])

    axis = 2
    out_shape = get_outshape(x, axis)
    u = uint16_to_uint8(x, axis=axis, msb_first=False).to(torch.int32)
    assert u.shape == out_shape
    assert torch.all(u[..., 0] + u[..., 1]*2**8 == x_32[..., 0])
    u = uint16_to_uint8(x, axis=axis, msb_first=True).to(torch.int32)
    assert u.shape == out_shape
    assert torch.all(u[..., 1] + u[..., 0]*2**8 == x_32[..., 0])