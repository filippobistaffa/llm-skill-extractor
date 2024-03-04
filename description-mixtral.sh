#!/bin/bash
#SBATCH --job-name=llama-cpp-description-mixtral
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --output=description-mixtral-%j.out
#SBATCH --error=description-mixtral-%j.err

HOSTNAME=$(hostname)

if [ "$HOSTNAME" == "vega.iiia.csic.es" ]
then
    spack load --first py-pandas
elif [ "$HOSTNAME" == "login*" ]
then
    module load pandas
fi

python3 description.py --model "llama.cpp/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf" --format "[INST] {} [/INST]" --seed $RANDOM
