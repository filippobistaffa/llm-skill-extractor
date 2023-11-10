#!/bin/bash
#SBATCH --job-name=llama-cpp-description-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=description-gpu.out
#SBATCH --error=description-gpu.err

if [ "$USER" == "filippo.bistaffa" ]
then
    spack load --first py-pandas
else
    module load python/3.9.9
fi

srun python3 description.py --n-gpu-layers 40
