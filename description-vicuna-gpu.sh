#!/bin/bash
#SBATCH --job-name=llama-cpp-description-vicuna-gpu
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=description-vicuna-gpu-%j.out
#SBATCH --error=description-vicuna-gpu-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

spack load cuda@11.8.0
spack load --first py-pandas

srun python3 description-vicuna.py --n-gpu-layers 32
