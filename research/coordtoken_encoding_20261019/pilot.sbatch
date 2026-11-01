#!/bin/bash
#SBATCH --job-name=ct_encode_check
#SBATCH --partition=research8
#SBATCH --account=research
#SBATCH --qos=researcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/coordtoken_encoding_20261019/pilot-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export PYTHONPATH="$PWD/research/coordtoken_encoding_20261019/source/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/coordtoken_encoding_20261019/pilot.py
