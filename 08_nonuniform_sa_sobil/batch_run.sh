#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --job-name=gosox
#SBATCH --partition=shas
#SBATCH --qos=condo
#SBATCH --account=ucb-summit-jrk
#SBATCH --time=0-00:100:00
#SBATCH --mail-user=kravitsjacob@gmail.com
#SBATCH --mail-type=END

module purge
source /curc/sw/anaconda3/2019.07/bin/activate
conda activate grid_optimization
python -u main.py '/scratch/summit/jakr3868/Water_OPF_Non-Uniform_Sobil_V3_io/' 48