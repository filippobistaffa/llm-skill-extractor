#!/bin/bash
#SBATCH --job-name=llama-cpp-label
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label.out
#SBATCH --error=label.err

if [ "$USER" == "filippo.bistaffa" ]
then
    spack load --first py-pandas
else
    module load python/3.9.9
fi

srun python3 label.py
