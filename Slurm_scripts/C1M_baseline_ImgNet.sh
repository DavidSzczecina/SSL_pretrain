#!/bin/bash
#SBATCH --job-name=Clothing1M_baseline_ImageNet
#SBATCH --account=def-pfieguth
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --output=slurm_output_C1M/Clothing1M_baseline_ImageNet_%j.out
#SBATCH --error=slurm_output_C1M/Clothing1M_baseline_ImageNet_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL

set -euo pipefail

# --- Env ---
module load python
source ../envs/ssl_env/bin/activate
cd ..


echo "Extracting Clothing1M dataset into /tmp..."

# --- Paths ---
META_DIR=/home/dszczeci/projects/def-pfieguth/dszczeci/SSL_pretrain/data/Clothing-1M/annotations
TARS_DIR="/home/dszczeci/projects/def-pfieguth/dszczeci/SSL_pretrain/data/Clothing-1M/images"

DATA_TMP="${SLURM_TMPDIR:-/tmp}/${USER:-$(whoami)}_clothing1m_${SLURM_JOB_ID:-$$}"
mkdir -p "$DATA_TMP"
trap 'rm -rf "$DATA_TMP" || true' EXIT
export TMPDIR="$DATA_TMP"

# Ensure cleanup
cleanup() { rm -rf "$DATA_TMP" || true; }
trap cleanup EXIT

# Iterate through all 10 tar files (0.tar, 1.tar, ..., 9.tar)
for i in {0..9}; do
  TAR_PATH="$TARS_DIR/${i}.tar"
  echo "[INFO] Unpacking $TAR_PATH ..."
  start=$(date +%s)
  tar -xf "$TAR_PATH" -C "$DATA_TMP"
  end=$(date +%s)
  runtime=$(( end - start ))
  echo "[INFO] Unpacked ${i}.tar in ${runtime}s"
done

echo "[INFO] Extraction complete. Size:"
du -sh "$DATA_TMP" || true




EPOCHS_SUP=15
SEEDS=(5)

# BASELINE: Supervised training on noisy labels (no SSL pretrain)
echo ">>> Running BASELINE supervised experiments..."
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="baseline-C1M_supE-${EPOCHS_SUP}_ImageNet_s-${SEED}"
    echo "[BASELINE] SEED=${SEED}"

    python ssl_C1M_experiment.py \
      --mode train_eval \
      --seed "${SEED}" \
      --workers 4 \
      --images_dir "$DATA_TMP" \
      --meta_dir "$META_DIR" \
      --epochs "${EPOCHS_SUP}" \
      --exp-name "${EXP_NAME}" \
      --imagenet-pretrain \
      --save-every 1

    echo ">>> Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
done



echo "[INFO] Job complete."