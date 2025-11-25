#!/bin/bash
#SBATCH --job-name=C100_baseline_E100
#SBATCH --account=def-pfieguth
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/C100_baseline_E100.out
#SBATCH --error=slurm_output/C100_baseline_E100.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Activate your virtual environment and load required modules
#source ../envs/env/bin/activate
#module load python

# Change directory to your project folder (adjust the path as needed)
cd ..



# Common experiment variables

DATASET="cifar100"
EPOCHS_SUP=100            # supervised fine-tuning epochs
NOISE_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
NOISE_RATES=(0.6)
SEEDS=(2 3 4 5)




# BASELINE: Supervised training on noisy labels (no SSL pretrain)


echo ">>> Running BASELINE supervised experiments..."
for SEED in "${SEEDS[@]}"; do
  for NOISE in "${NOISE_RATES[@]}"; do
    EXP_NAME="baseline_${DATASET}_supE-${EPOCHS_SUP}_noise-${NOISE}_s-${SEED}"
    echo "[BASELINE] SEED=${SEED} NOISE=${NOISE}"

    python ssl_cifar_experiment.py \
      --dataset "${DATASET}" \
      --results-root results_c100_long \
      --mode train_eval \
      --seed "${SEED}" \
      --epochs "${EPOCHS_SUP}" \
      --noise-rate "${NOISE}" \
      --exp-name "${EXP_NAME}"

    echo ">>> Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
  done
done



echo "===== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ====="









