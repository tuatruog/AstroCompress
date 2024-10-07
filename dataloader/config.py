import os


class ConfigConstants:
    DATA_PROFILE_FULL = 'full'
    DATA_PROFILE_TINY = 'tiny'

    # Use 'tiny' for tiny dataset for fast experimenting, change to 'full' for real training
    DATA_PROFILE = DATA_PROFILE_TINY
    # Where local data is stored:
    LOCAL_DATA_ROOT = './'
    # Change to your personal local hugging face cache directory
    DATA_CACHE_DIR = os.path.join(LOCAL_DATA_ROOT, ".cache")

    # Split ratio of data
    DATA_SPLIT_RATIO = 0.85
