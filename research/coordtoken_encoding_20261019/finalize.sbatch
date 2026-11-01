#!/bin/bash
#SBATCH --job-name=ct_encode_verify
#SBATCH --partition=research_cpu
#SBATCH --account=research
#SBATCH --qos=researcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/coordtoken_encoding_20261019/verify-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export PYTHONPATH="$PWD/research/coordtoken_encoding_20261019/source/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/coordtoken_encoding_20261019/finalize.py "$1"
