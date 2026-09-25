#!/bin/bash
#SBATCH --job-name=nabla_inchikey_audit
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
set -euo pipefail
AUDIT_ROOT="${COORDTOKEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
AUDIT_PYTHON="${COORDTOKEN_PYTHON:-python}"
AUDIT_OUTPUT="${COORDTOKEN_OUTPUT:-$AUDIT_ROOT/logs/inchikey_overlap/nabla_scaffolds_structures_190m}"
COORDTOKEN_SOURCE_ROOT="${COORDTOKEN_SOURCE_ROOT:-/mnt/weka/fgeikyan/fsq}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$AUDIT_ROOT"
exec "$AUDIT_PYTHON" -u scripts/audit_inchikey_overlap.py \
  --train "$COORDTOKEN_SOURCE_ROOT/merged_train.csv" \
  --reference "$COORDTOKEN_SOURCE_ROOT/test/nablaDFT_scaffolds.csv" \
  --reference "$COORDTOKEN_SOURCE_ROOT/test/nablaDFT_structures.csv" \
  --output "$AUDIT_OUTPUT" \
  --workers "${SLURM_CPUS_PER_TASK:-32}" --benchmark --resume
