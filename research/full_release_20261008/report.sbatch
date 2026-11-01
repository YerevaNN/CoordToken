#!/bin/bash
#SBATCH --job-name=coordtoken_report_oct8
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/full_release_20261008/report-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/full_release_20261008/report_status.py
