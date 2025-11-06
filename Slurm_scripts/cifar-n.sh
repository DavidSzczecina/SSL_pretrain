#!/bin/bash
#SBATCH --job-name=Cifar-n_s5
#SBATCH --account=def-pfieguth
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=slurm_output/Cifar-n.out
#SBATCH --error=slurm_output/Cifar-n.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Activate your virtual environment and load required modules
#source ../envs/env/bin/activate
#module load python

# Change directory to your project folder (adjust the path as needed)
cd ..



# Common experiment variables

DATASET="cifar-10n"
EPOCHS_SUP=10             # supervised fine-tuning epochs
EPOCHS_PRE=(5 10 25 50 75 100) # pretraining epochs to compare
SEEDS=(1 2 3 4 5)



# BASELINE: Supervised training on noisy labels (no SSL pretrain)


# CIFAR-10n --cifar10n-label-type aggre_label

echo ">>> Running BASELINE supervised experiments..."
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="baseline_${DATASET}-Aggre_supE-${EPOCHS_SUP}_s-${SEED}"
    echo "[BASELINE] SEED=${SEED}"

    python ssl_cifar_experiment.py \
      --dataset "${DATASET}" \
      --cifar10n-label-type aggre_label \
      --mode train_eval \
      --seed "${SEED}" \
      --epochs "${EPOCHS_SUP}" \
      --exp-name "${EXP_NAME}"

    echo ">>> Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
done


 
# CIFAR-10n --cifar10n-label-type worse_label
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="baseline_${DATASET}-Worse_supE-${EPOCHS_SUP}_s-${SEED}"
    echo "[BASELINE] SEED=${SEED}"

    python ssl_cifar_experiment.py \
      --dataset "${DATASET}" \
      --cifar10n-label-type worse_label \
      --mode train_eval \
      --seed "${SEED}" \
      --epochs "${EPOCHS_SUP}" \
      --exp-name "${EXP_NAME}"

    echo ">>> Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
done


DATASET="cifar-100n"
# CIFAR-100n
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="baseline_${DATASET}_supE-${EPOCHS_SUP}_s-${SEED}"
    echo "[BASELINE] SEED=${SEED}"

    python ssl_cifar_experiment.py \
      --dataset "${DATASET}" \
      --mode train_eval \
      --seed "${SEED}" \
      --epochs "${EPOCHS_SUP}" \
      --exp-name "${EXP_NAME}"

    echo ">>> Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
done







      
      

      



DATASET="cifar-10n"
# FINE-TUNE FROM PRETRAINED ENCODERS

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
        --pretrained-encoder-path "simclr_cifar10_e${PRE_E}_s${SEED}.pth"
    
      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
  done
done

DATASET="cifar-10n"
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
        --pretrained-encoder-path "simclr_cifar10_e${PRE_E}_s${SEED}.pth"
    
      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
  done
done



DATASET="cifar-100n"
echo ">>> Running FINE-TUNING from pretrained encoders..."
for SEED in "${SEEDS[@]}"; do
  for PRE_E in "${EPOCHS_PRE[@]}"; do
  
      EXP_NAME="simclr_${DATASET}_preE-${PRE_E}_supE-${EPOCHS_SUP}_s-${SEED}"
      echo "[FINETUNE] SEED=${SEED} PRETRAIN_EPOCHS=${PRE_E} Supervised_Epochs=${EPOCHS_SUP}"
    
      python ssl_cifar_experiment.py \
        --dataset "${DATASET}" \
        --mode train_eval \
        --seed "${SEED}" \
        --epochs "${EPOCHS_SUP}" \
        --exp-name "${EXP_NAME}" \
        --pretrained-encoder-path "simclr_cifar100_e${PRE_E}_s${SEED}.pth"
    
      echo ">>> Finished fine-tuning: ${EXP_NAME}"
      echo "---------------------------------------------"
  done
done




echo "===== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ====="






