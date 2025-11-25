#!/bin/bash
#SBATCH --job-name=C100_supervised_s1
#SBATCH --account=def-pfieguth
#SBATCH --time=47:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/C100_supervised_s1.out
#SBATCH --error=slurm_output/C100_supervised_s1.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Activate your virtual environment and load required modules
#source ../envs/env/bin/activate
#module load python

# Change directory to your project folder (adjust the path as needed)
cd ..



# Common experiment variables

DATASET="cifar100"
EPOCHS_SUP=100             # supervised fine-tuning epochs
EPOCHS_PRE=(10 25 50 100) # pretraining epochs to compare
NOISE_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
NOISE_RATES=(0.0 0.2 0.4 0.6 0.8 0.9)
SEEDS=(1)



# FINE-TUNE FROM PRETRAINED ENCODERS

echo ">>> Running FINE-TUNING from pretrained encoders..."
for SEED in "${SEEDS[@]}"; do
  for PRE_E in "${EPOCHS_PRE[@]}"; do
    for NOISE in "${NOISE_RATES[@]}"; do

      EXP_NAME="simclr_${DATASET}_preE-${PRE_E}_supE-${EPOCHS_SUP}_noise-${NOISE}_s-${SEED}"
      echo "[FINETUNE] SEED=${SEED} PRETRAIN_EPOCHS=${PRE_E} NOISE=${NOISE} Supervised_Epochs=${EPOCHS_SUP}"

      python ssl_cifar_experiment.py \
        --dataset "${DATASET}" \
        --results-root results_C100_long \
        --mode train_eval \
        --seed "${SEED}" \
        --epochs "${EPOCHS_SUP}" \
        --noise-rate "${NOISE}" \
        --exp-name "${EXP_NAME}" \
        --pretrained-encoder-path "simclr_${DATASET}_e${PRE_E}_s${SEED}.pth"

      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
    done
  done
done


echo "===== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ====="






