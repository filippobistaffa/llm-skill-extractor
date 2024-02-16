#!/bin/bash
#SBATCH --job-name=llama-cpp-label-mixtral
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label-mixtral-%j.out
#SBATCH --error=label-mixtral-%j.err

spack load --first py-pandas

cmd=$(mktemp ./tempfile.XXXXXX.sh)
srun python3 label.py --model "llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf" --format "[INST] {} [/INST]" --seed $RANDOM --cmd $cmd
bash $cmd
