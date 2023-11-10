#!/bin/bash
#SBATCH --job-name=llama-cpp-description-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --output=description-gpu.out
#SBATCH --error=description-gpu.err

module load python/3.9.9

srun python3 description.py --threads 20 --n-gpu-layers 40
