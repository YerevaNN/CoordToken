#!/bin/bash
#SBATCH --job-name=ct_nb22_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/fs1/filya/CoordToken/logs/nabla_b200_20261023/eval22-%j.log
set -euo pipefail
umask 0002
cd /mnt/fs1/filya/CoordToken
export PYTHONPATH=/mnt/fs1/filya/envs/b200_snapshot/extra:/mnt/fs1/filya/envs/b200_snapshot/site-packages
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 NABLA_WIDTH=1024
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>logs/nabla_b200_20261023/w1024/epoch_22/evaluation.lock
flock -n 9
if test -f logs/nabla_b200_20261023/w1024/epoch_22/evaluation/result.json; then
    echo 'Evaluation result already exists; refusing to overwrite.'
    exit 1
fi
/usr/bin/python3 -u - <<'PY'
import torch
assert torch.cuda.is_available()
assert 'B200' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print('Evaluation GPU:', torch.cuda.get_device_name(0), flush=True)
PY
/usr/bin/python3 -u research/nabla_b200_20261023/evaluate.py --endpoint epoch_22
