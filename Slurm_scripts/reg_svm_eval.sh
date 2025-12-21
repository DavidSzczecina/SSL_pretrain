#!/bin/bash
#SBATCH --job-name=C10_ssl_sklearn
#SBATCH --account=def-pfieguth
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/C10_ssl_sklearn.out
#SBATCH --error=slurm_output/C10_ssl_sklearn.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Change directory to your project folder (adjust the path as needed)
cd ..

# -----------------------------
# Common experiment variables
# -----------------------------
DATASET="cifar10"

EPOCHS_PRE=(10 25 50 100)    # pretraining epochs to compare
NOISE_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
NOISE_RATES=(0.0 0.2 0.4 0.6 0.8)
SEEDS=(1 2 3 4 5)
SEEDS=(1)   # adjust if you want all 5 seeds

echo ">>> Running SSL feature regression + SVM experiments..."

for SEED in "${SEEDS[@]}"; do
  for PRE_E in "${EPOCHS_PRE[@]}"; do
    for NOISE in "${NOISE_RATES[@]}"; do

      EXP_NAME="simclr_${DATASET}_skl_preE-${PRE_E}_noise-${NOISE}_s-${SEED}"
      echo "[SSL_SKLEARN] SEED=${SEED} PRETRAIN_EPOCHS=${PRE_E} NOISE=${NOISE}"

      python ssl_cifar_experiment.py \
        --dataset "${DATASET}" \
        --mode ssl_sklearn \
        --seed "${SEED}" \
        --noise-rate "${NOISE}" \
        --exp-name "${EXP_NAME}" \
        --pretrain-name "simclr" \
        --pretrain-epochs "${PRE_E}" \
        --pretrained-encoder-path "simclr_${DATASET}_e${PRE_E}_s${SEED}.pth"

      echo ">>> Finished SSL-sklearn run: ${EXP_NAME}"
      echo "---------------------------------------------"
    done
  done
done

echo "===== ALL SSL-SKLEARN EXPERIMENTS COMPLETED SUCCESSFULLY ====="
