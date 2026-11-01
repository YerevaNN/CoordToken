#!/bin/bash
#SBATCH --job-name=coordtoken_bench_oct8
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/full_release_20261008/benchmark-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/coordtoken_oct8_${SLURM_JOB_ID}
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/full_release_20261008/frozen/benchmark.py
