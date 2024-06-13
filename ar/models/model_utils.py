"""
Utils for model definition / creation.
Based on https://github.com/yang-song/score_sde_pytorch/blob/main/models/utils.py
"""
import torch

_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    """Create the score model.
    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.

    """
    model_name = config.model.name
    model = get_model(model_name)(config)
    # Might use accelerate later for multi-gpu training.
    # model = model.to(config.device)
    # model = torch.nn.DataParallel(model)
    return model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


# For astro compression specifically:
def get_num_input_channels(data_spec, split_bits_axis):
    """
    Args:
        data_spec: str, e.g., 'sdss'.
        split_uint16_to_uint8_axis: int or None, the axis along which to split uint16 to uint8.
    """
    from utils.data_loader import DATASET_NUM_CHANNELS
    ch = DATASET_NUM_CHANNELS[data_spec]
    assert split_bits_axis in (None, 0, 1, 2)
    if split_bits_axis == None:
        # No conversion to uint8
        pass
    elif split_bits_axis == 0:
        # Splits along the channel dim; doubling.
        ch *= 2
    else:
        # Splits along spatial (H or W) dim.
        pass
    return ch
