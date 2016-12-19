#!/bin/bash

#SBATCH --job-name=cudabench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=result.out
#SBATCH --error=log.err

source /urs/share/Modules/init/bash
unset MODULEPATH
module use /lustre/usr/modulfiles/pi
module purge
module load gcc cuda

./a.out
