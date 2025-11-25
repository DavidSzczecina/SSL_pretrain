#!/bin/bash
#SBATCH --job-name=C10-n_supervised_s5
#SBATCH --account=def-pfieguth
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm_output/C10-n_supervised_s5.out
#SBATCH --error=slurm_output/C10-n_supervised_s5.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Activate your virtual environment and load required modules
#source ../envs/env/bin/activate
#module load python

# Change directory to your project folder (adjust the path as needed)
cd ..



# Common experiment variables

DATASET="cifar-10n"
PRETRAIN_DATASET="cifar10"
EPOCHS_SUP=10             # supervised fine-tuning epochs
EPOCHS_PRE=(5 10 25 50 75 100) # pretraining epochs to compare
SEEDS=(1 2 3 4 5)





# FINE-TUNE FROM PRETRAINED ENCODERS

# CIFAR-10n --cifar10n-label-type aggre_label
echo ">>> Running FINE-TUNING from pretrained encoders..."
for SEED in "${SEEDS[@]}"; do
  for PRE_E in "${EPOCHS_PRE[@]}"; do
      EXP_NAME="simclr_${DATASET}-Aggre_preE-${PRE_E}_supE-${EPOCHS_SUP}_s-${SEED}"
      echo "[FINETUNE] SEED=${SEED} PRETRAIN_EPOCHS=${PRE_E} Supervised_Epochs=${EPOCHS_SUP}"

      python ssl_cifar_experiment.py \
        --dataset "${DATASET}" \
        --cifar10n-label-type aggre_label \
        --mode train_eval \
        --seed "${SEED}" \
        --epochs "${EPOCHS_SUP}" \
        --exp-name "${EXP_NAME}" \
        --pretrained-encoder-path "simclr_${PRETRAIN_DATASET}_e${PRE_E}_s${SEED}.pth"

      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
  done
done



# CIFAR-10n --cifar10n-label-type worse_label
echo ">>> Running FINE-TUNING from pretrained encoders..."
for SEED in "${SEEDS[@]}"; do
  for PRE_E in "${EPOCHS_PRE[@]}"; do
      EXP_NAME="simclr_${DATASET}-Worse_preE-${PRE_E}_supE-${EPOCHS_SUP}_s-${SEED}"
      echo "[FINETUNE] SEED=${SEED} PRETRAIN_EPOCHS=${PRE_E} Supervised_Epochs=${EPOCHS_SUP}"

      python ssl_cifar_experiment.py \
        --dataset "${DATASET}" \
        --cifar10n-label-type worse_label \
        --mode train_eval \
        --seed "${SEED}" \
        --epochs "${EPOCHS_SUP}" \
        --exp-name "${EXP_NAME}" \
        --pretrained-encoder-path "simclr_${PRETRAIN_DATASET}_e${PRE_E}_s${SEED}.pth"

      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
  done
done


echo "===== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ====="






