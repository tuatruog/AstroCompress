import os

# Configurations for data loading.
# Update this according to your own setup; no need to commit local changes.

# Use 'tiny' for tiny dataset for fast experimenting, change to 'full' for real training
DATA_PROFILE = 'full'
# Where local data is stored:
LOCAL_DATA_ROOT = './'
# Change to your personal local hugging face cache directory
DATA_CACHE_DIR = os.path.join(LOCAL_DATA_ROOT, ".cache")