#!/bin/bash

#SBATCH --job-name=cudabench
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH --output=k80.out
#SBATCH --error=log.err

source /usr/share/Modules/init/bash
unset MODULEPATH
module use /lustre/usr/modulefiles/pi
module purge
module load gcc cuda

./mat
