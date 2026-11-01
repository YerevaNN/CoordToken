#!/bin/bash
#SBATCH --job-name=coordtoken_prepare_oct8
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/full_release_20261008/prepare-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/full_release_20261008/prepare.py --release /mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_final --output logs/full_release_20261008/packed --workers 64
