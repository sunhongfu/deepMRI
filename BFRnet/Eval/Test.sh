#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o BFRnet_New2.txt
#SBATCH -e Error.txt
#SBATCH --job-name=XYZ_BFR
#SBATCH -c 3
#SBATCH --mem=100000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


module purge
module load gnu7
module load cuda/10.0.130
module load anaconda/3.6
module load mvapich2
module load matlab/R2019a

srun matlab -r "BFRnet_demo"