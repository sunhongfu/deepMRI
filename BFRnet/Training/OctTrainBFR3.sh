#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o BFR64_24BS_45Epo_NewHCmix_2GPU.txt
#SBATCH -e BFRError64_24BS_45Epo_NewHCmix.txt
#SBATCH --job-name=XYZ_BFR
#SBATCH -c 3
#SBATCH --mem=30000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:2


module purge
module load gnu7
module load cuda/10.0.130
module load anaconda/3.6
module load mvapich2
module load matlab/R2019a

srun matlab -r "TrainOctNet130BFR3"