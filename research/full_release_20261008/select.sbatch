#!/bin/bash
#SBATCH --job-name=coordtoken_select_oct8
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/full_release_20261008/select-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/full_release_20261008/frozen/choose_model.py
