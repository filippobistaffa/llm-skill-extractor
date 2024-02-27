#!/bin/bash
#SBATCH --job-name=llama-cpp-description-vicuna
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --output=description-vicuna-%j.out
#SBATCH --error=description-vicuna-%j.err

spack load --first py-pandas

python3 description.py --model "llama.cpp/models/vicuna-13b-v1.5-16k.Q4_K_M.gguf" --format "USER: {}\nASSISTANT:" --seed $RANDOM
