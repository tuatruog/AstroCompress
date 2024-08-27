template = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output=/extra/ucibdl1/shared/data/astrocomp/job-log/{config_id}.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tuannt2@uci.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=ava_m.p

source /home/tuannt2/.bashrc     # the ICS default .bashrc makes 'module' available
module load conda
module load slurm

# for Tensorflow
export TF_CPP_MIN_LOG_LEVEL=2   # ignore TF libpng warnings: https://github.com/tensorflow/tensorflow/issues/31870

source activate astro-compression
conda env list

cd /home/tuannt2/projects/astro-compression

{srun_command}
"""