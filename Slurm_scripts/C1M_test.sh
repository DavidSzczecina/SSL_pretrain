#!/bin/bash
#SBATCH --job-name=Clothing1M_test
#SBATCH --account=def-pfieguth
#SBATCH --time=0:59:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output_C1M/Clothing1M_test_%j.out
#SBATCH --error=slurm_output_C1M/Clothing1M_test_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


set -euo pipefail

# --- Env ---
#module load python
#source ../envs/env/bin/activate
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




EPOCHS_SUP=1
SEEDS=(1)

# BASELINE: Supervised training on noisy labels (no SSL pretrain)
echo ">>> Running BASELINE supervised experiments..."
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="baseline-C1M_supE-${EPOCHS_SUP}_s-${SEED}"
    echo "[BASELINE] SEED=${SEED}"

    python ssl_C1M_experiment.py \
      --mode train_eval \
      --seed "${SEED}" \
      --images_dir "$DATA_TMP" \
      --meta_dir "$META_DIR" \
      --epochs "${EPOCHS_SUP}" \
      --exp-name "${EXP_NAME}"

    echo ">>> Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
done




SSL_EPOCHS=1
SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
    EXP_NAME="simclr_C1M_e${SSL_EPOCHS}_s${SEED}"
    echo "[SSL PRETRAIN] SEED=${SEED} EPOCHS=${SSL_EPOCHS}"
    
    python ssl_C1M_experiment.py \
    --mode pretrain \
    --pretrain-name simclr \
    --seed "${SEED}" \
    --images_dir "$DATA_TMP" \
    --meta_dir "$META_DIR" \
    --pretrain-epochs "${SSL_EPOCHS}" \
    --exp-name "${EXP_NAME}" \
    --save-pretrained-encoder "simclr_C1M_e${SSL_EPOCHS}_s${SEED}.pth" \
    
    echo ">>> Finished pretraining: ${EXP_NAME}"
    echo "---------------------------------------------"
done


EPOCHS_PRE=(1)

echo ">>> Running FINE-TUNING from pretrained encoders..."
for SEED in "${SEEDS[@]}"; do
  for PRE_E in "${EPOCHS_PRE[@]}"; do
      EXP_NAME="simclr_C1M_preE-${PRE_E}_supE-${EPOCHS_SUP}_s-${SEED}"
      echo "[FINETUNE] SEED=${SEED} PRETRAIN_EPOCHS=${PRE_E} Supervised_Epochs=${EPOCHS_SUP}"

      python ssl_C1M_experiment.py \
        --results-root results_C1M \
        --mode train_eval \
        --seed "${SEED}" \
        --images_dir "$DATA_TMP" \
        --meta_dir "$META_DIR" \
        --epochs "${EPOCHS_SUP}" \
        --exp-name "${EXP_NAME}" \
        --pretrained-encoder-path "simclr_C1M_e${PRE_E}_s${SEED}.pth"

      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
  done
done

echo "[INFO] Job complete."