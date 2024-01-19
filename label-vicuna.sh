#!/bin/bash
#SBATCH --job-name=llama-cpp-label-vicuna
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label-vicuna.out
#SBATCH --error=label-vicuna.err

spack load --first py-pandas

srun python3 label.py --seed $RANDOM
