#!/bin/bash
#SBATCH --job-name=simsiam_encoder
#SBATCH --account=def-pfieguth
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=slurm_output/simsiam_encoder.out
#SBATCH --error=slurm_output/simsiam_encoder.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL


# Activate your virtual environment and load required modules
#source ../envs/env/bin/activate
#module load python

# Change directory to your project folder (adjust the path as needed)
cd ..

# SELF-SUPERVISED PRETRAINING 
echo ">>> Running SSL PRETRAINING experiments..."





SSL_EPOCHS=100
DATASET=cifar10
SEED=1



EXP_NAME="simclr_${DATASET}_e${SSL_EPOCHS}_s${SEED}"
echo "[SSL PRETRAIN] SEED=${SEED} EPOCHS=${SSL_EPOCHS}"

python ssl_cifar_experiment.py \
--dataset "${DATASET}" \
--mode pretrain \
--pretrain-name simsiam \
--seed "${SEED}" \
--pretrain-epochs "${SSL_EPOCHS}" \
--exp-name "${EXP_NAME}" \
--save-pretrained-encoder "simsiam_${DATASET}_e${SSL_EPOCHS}_s${SEED}" \

echo ">>> Finished pretraining: ${EXP_NAME}"
echo "---------------------------------------------"





SSL_EPOCHS=100
DATASET=cifar100
SEED=1

EXP_NAME="simclr_${DATASET}_e${SSL_EPOCHS}_s${SEED}"
echo "[SSL PRETRAIN] SEED=${SEED} EPOCHS=${SSL_EPOCHS}"

python ssl_cifar_experiment.py \
--dataset "${DATASET}" \
--mode pretrain \
--pretrain-name simsiam \
--seed "${SEED}" \
--pretrain-epochs "${SSL_EPOCHS}" \
--exp-name "${EXP_NAME}" \
--save-pretrained-encoder "simsiam_${DATASET}_e${SSL_EPOCHS}_s${SEED}" \

echo ">>> Finished pretraining: ${EXP_NAME}"
echo "---------------------------------------------"







echo "===== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ====="






