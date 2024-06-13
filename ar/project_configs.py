from pathlib import Path

PROJECT_DIR = Path("/home/yiboyang/projects/astro-compression")  # Set this to current dir.
SETUP_SCRIPT_PATH = PROJECT_DIR / "ar" / "setup.sh"  # Init script run by Slurm to configure python etc.
SLURM_JOBS_DIR = PROJECT_DIR / "slurm_jobs"  # Where to store slurm job scripts for reference.

TORCH_DATASETS_DIR = "/extra/ucibdl0/shared/data/torch_datasets"
EXPERIMENTS_DIR = "/extra/ucibdl1/shared/projects/astrocomp_expms/"  # Where to store experiment artifacts.

# Conventions for names/tags of dirs/groups used in training.
TRAIN_COLLECTION = "train"
EVAL_COLLECTION = "eval"
CHECKPOINTS_DIR_NAME = "checkpoints"
