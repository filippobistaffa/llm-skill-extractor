#!/bin/bash
#SBATCH --job-name=llama-cpp-label-mixtral
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label-mixtral-%j.out
#SBATCH --error=label-mixtral-%j.err

spack load --first py-pandas

cmd=label-mixtral-$SLURM_JOB_ID.cmd
srun python3 label.py --model "llama.cpp/models/gemma-7b-it-Q4_K_M.gguf" --format "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model" --seed $RANDOM --cmd $cmd
bash $cmd
