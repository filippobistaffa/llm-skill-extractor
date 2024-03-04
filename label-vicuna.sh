#!/bin/bash
#SBATCH --job-name=llama-cpp-label-vicuna
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label-vicuna-%j.out
#SBATCH --error=label-vicuna-%j.err

HOSTNAME=$(hostname)

if [ "$HOSTNAME" == "vega.iiia.csic.es" ]
then
    spack load --first py-pandas
elif [ "$HOSTNAME" == "login*" ]
then
    module load pandas
fi

cmd=label-vicuna-$SLURM_JOB_ID.cmd
srun python3 label.py --model "llama.cpp/models/vicuna-13b-v1.5-16k.Q4_K_M.gguf" --format "USER: {}\nASSISTANT:" --seed $RANDOM --cmd $cmd
bash $cmd
