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

srun python3 description.py --model "llama.cpp/models/vicuna-13b-v1.5-16k.Q4_K_M.gguf" --format "USER: {}\nASSISTANT:" --seed $RANDOM --n-gpu-layers 32
