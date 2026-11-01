#!/bin/bash
#SBATCH --job-name=coordtoken_pilots_oct8
#SBATCH --partition=research8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/full_release_20261008/pilots-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/coordtoken_oct8_${SLURM_JOB_ID}
PYTHON=/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python
"$PYTHON" -u research/full_release_20261008/frozen/gpu_check.py
"$PYTHON" -u research/full_release_20261008/frozen/run_pilots.py
