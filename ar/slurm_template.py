template = """#!/bin/bash
#SBATCH --job-name={job_name}         # create a short name for your job
#SBATCH --array=0-{last_hid}
#SBATCH --output={slurm_jobs_dir}/%A_%a.out
#SBATCH --error={slurm_jobs_dir}/%A_%a.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (= mem-per-cpu * cpus-per-task)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --partition=ava_m.p              # submit to the Mandt lab queue (use ava_s.p to use Sameer's queue)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yiboyang@uci.edu    # your email address to receive notifications
## Unused options below
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=all          # send email on job start, end and fault
##SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
##SBATCH --mem-per-cpu=4G         # memory per cpu-core

# >>>>>>>>>>>>>>>>>>>>>> Setup >>>>>>>>>>>>>>>>>>>>>
# module purge
source /home/yiboyang/.bashrc     # the ICS default .bashrc makes 'module' available
module load cuda/11.2
module load slurm

# for Tensorflow
export TF_CPP_MIN_LOG_LEVEL=2   # ignore TF libpng warnings: https://github.com/tensorflow/tensorflow/issues/31870
export TFDS_DATA_DIR='/extra/ucibdl0/yiboyang/data/tensorflow_datasets/'  # use the group's data server; home storage low
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Prevent TF from grabbing all GPU mem immediately. https://stackoverflow.com/a/55541385

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/yiboyang/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yiboyang/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yiboyang/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/yiboyang/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda deactivate # https://github.com/conda/conda/issues/9392 to ensure correct python path is prepended to PATH with `conda activate`

# for Rust
export PATH="$HOME/.cargo/bin:$PATH"

# for my other custom binaries (e.g., magick)
export PATH="$HOME/software/bin:$PATH"

# Project specific:
cd {project_dir}
. {setup_script_path}
# To make files created by the job automatically shared with ucibdl.
umask 007

# <<<<<<<<<<<<<<<<<<<<<< Setup <<<<<<<<<<<<<<<<<<<<<<<<

{srun_command}

"""

