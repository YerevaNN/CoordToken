#!/bin/bash
#SBATCH --job-name=ct_encode
#SBATCH --partition=research8
#SBATCH --account=research
#SBATCH --qos=researcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/coordtoken_encoding_20261019/encode-%A_%a.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export PYTHONPATH="$PWD/research/coordtoken_encoding_20261019/source/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/coordtoken_encoding_20261019/worker.py "$1" "$2" "$SLURM_ARRAY_TASK_ID"
