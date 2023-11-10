#!/bin/bash
#SBATCH --job-name=llama-cpp-label
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=label.out
#SBATCH --error=label.err

module load python/3.9.9

srun python3 label.py
