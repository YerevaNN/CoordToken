#!/bin/bash
#SBATCH --job-name=nabla_all_inchikey_audit
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COORDTOKEN_ROOT="${COORDTOKEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
COORDTOKEN_PYTHON="${COORDTOKEN_PYTHON:-python}"
COORDTOKEN_SOURCE_ROOT="${COORDTOKEN_SOURCE_ROOT:-/mnt/weka/fgeikyan/fsq}"
cd "$COORDTOKEN_ROOT"
exec "$COORDTOKEN_PYTHON" -u scripts/audit_inchikey_overlap.py \
 --train "$COORDTOKEN_SOURCE_ROOT/merged_train.csv" \
 --reference "$COORDTOKEN_SOURCE_ROOT/val/nablaDFT.csv" \
 --reference "$COORDTOKEN_SOURCE_ROOT/test/nablaDFT_conformations.csv" \
 --reference "$COORDTOKEN_SOURCE_ROOT/test/nablaDFT_scaffolds.csv" \
 --reference "$COORDTOKEN_SOURCE_ROOT/test/nablaDFT_structures.csv" \
 --output "${COORDTOKEN_OUTPUT:-logs/inchikey_overlap/nabla_all_val_test_190m}" \
 --workers "${SLURM_CPUS_PER_TASK:-32}" --benchmark --resume
