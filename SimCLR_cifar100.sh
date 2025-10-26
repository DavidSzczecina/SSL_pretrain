#!/bin/bash
#SBATCH --job-name=SimCLR_C100
#SBATCH --account=def-pfieguth
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=slurm_output/SimCLR_C100.out
#SBATCH --error=slurm_output/SimCLR_C100.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Activate your virtual environment and load required modules
#source ../envs/env/bin/activate
#module load python

# Change directory to your project folder (adjust the path as needed)
#cd ..



#SEEDS=(1 2 3 4 5)
SEEDS=(1)

NOISE_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# baseline experiments
for SEED in "${SEEDS[@]}"; do
  for NOISE in "${NOISE_RATES[@]}"; do
    EXP_NAME="baseline_c100_noise-${NOISE}_s${SEED}"
    echo "Running baseline: SEED=${SEED}, NOISE=${NOISE}, EXP_NAME=${EXP_NAME}"

    python ssl_cifar_experiment.py \
      --dataset cifar100 \
      --mode baseline \
      --seed "${SEED}" \
      --epochs 10 \
      --noise-rate "${NOISE}" \
      --device cuda \
      --exp-name "${EXP_NAME}"

    echo "Finished baseline: ${EXP_NAME}"
    echo "---------------------------------------------"
  done
done


#SEEDS=(1 2 3 4 5)
SEEDS=(1)
EPOCHS=(1 5 10 15 20 25 50)

#pretrain encoders

for SEED in "${SEEDS[@]}"; do
  for E in "${EPOCHS[@]}"; do
    EXP_NAME="simclr_c100_e${E}_s${SEED}"
    echo "Running SimCLR pretrain: SEED=${SEED}, PRETRAIN_EPOCHS=${E}, EXP_NAME=${EXP_NAME}"

    python ssl_cifar_experiment.py \
      --dataset cifar100 \
      --mode simclr_then_finetune \
      --seed "${SEED}" \
      --epochs 10 \
      --pretrain-epochs "${E}" \
      --exp-name "${EXP_NAME}" \
      --save-pretrained-encoder "simclr_c100_e${E}_s${SEED}"

    echo "Finished SimCLR pretrain+FT: ${EXP_NAME}"
    echo "---------------------------------------------"
  done
done



#SEEDS=(1 2 3 4 5)
SEEDS=(1)
EPOCHS=(1 5 10 15 20 25 50)
NOISE_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# finetune from pretrained encoders
for SEED in "${SEEDS[@]}"; do
  for E in "${EPOCHS[@]}"; do
    for NOISE in "${NOISE_RATES[@]}"; do

        EXP_NAME="simclr_c100_e${E}_s${SEED}_noise-${NOISE}"
        echo "Running finetuning from pretrained: SEED=${SEED}, PRETRAIN_EPOCHS=${E}, noise=${NOISE} EXP_NAME=${EXP_NAME}"

        python ssl_cifar_experiment.py \
        --dataset cifar100 \
        --mode finetune_from_pretrained \
        --seed "${SEED}" \
        --epochs 10 \
        --noise-rate "${NOISE}"\
        --exp-name "${EXP_NAME}" \
        --pretrained-encoder-path "pretrained_encoders/simclr_c100_e${E}_s${SEED}"

        echo "Finished SimCLR pretrain+FT: ${EXP_NAME}"
        echo "---------------------------------------------"
    done
  done
done


echo "All experiments submitted!"