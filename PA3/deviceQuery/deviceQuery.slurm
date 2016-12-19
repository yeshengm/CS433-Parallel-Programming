#!/bin/sh
#SBATCH --job-name=GTCP_UMA
#SBATCH --partition=<node_type>
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@mail.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=1:00:00


MODULEPATH=/lustre/utility/modulefiles:$MODULEPATH
module purge
module load cuda/8.0

./deviceQuery

