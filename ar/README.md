Custom code for model training/evaluation with Slurm integration and experiment tracking.

Currently only implements PixelCNN++.

## Code structure
- `common`: library/template/util code for train/eval/sampling/etc.
- `models`: model definitions. Under heavy construction.
- `configs`: config files for running experiments. We use a `ml_collections.ConfigDict` to manage configurations, which is the input to the train/eval/etc. routines in `common`.

- `train.py`: driver script for running training.

Example commands (run from project root dir):
Local training run with the default config (useful for debugging):
```
python -m ar.train --config ar/configs/pcnn_2d_i8.py --config.training.n_steps 1000 --experiments_dir /tmp/my_expms --alsologtostderr
```

Example training / hyperparameter search via Slurm:
```
python -m ar.launch.py --main ar.train --config ar/configs/pcnn_2d_i8.py
```

(additional slurm options can be specified in --sargs, e.g., '--gres=gpu:2' to override the default and request 2 GPUs instead of 1)


Evaluating a trained model on a given dataset (local):
```
python -m ar.eval --workdir my_workdir --dataset lco
```
where `my_workdir` is created from training and may look like `expms/214971/wid=0-pixelcnn-data=lco-split_bits=0-model=pixelcnn-nr_resnet=5-nr_filters=160-nr_mix=12`.


## Setup
- Edit `project_configs.py` to point `project_dir` to this directory.
- Put bash initialization commands (e.g., conda setup) in `setup.sh`; this is expected by the slurm template.
- Edit `slurm_template.py` to use your email and add your own slurm initialization commands.

We can add these config files to .gitignore to avoid commiting user-specific changes.



## How to extend the code

- Define your custom model as a module in `models/` that implements `compute_loss` and optionally `evaluate` (these are expected by the training template code); give it a string identifier and register it with `model_utils.register_model`. See the `Model` class in [pixelcnn.py](models/pixelcnn.py) for example.
- Import the resulting module in `common/experiments.py`, like `from ar.models import pixelcnn, model1, model2`; this will make sure the model registration works correctly.
- Connect configs for the model to a config script; examples in `configs`.


## References

Code:

- https://github.com/yang-song/score_sde_pytorch (for training template and layout of project and config scripts)
- https://github.com/google-research/vdm (for offical VDM in jax, training template and project layout)


