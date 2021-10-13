#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --job-name=gosox
#SBATCH --partition=shas
#SBATCH --qos=condo
#SBATCH --account=ucb-summit-jrk
#SBATCH --time=0-00:15:00
#SBATCH --mail-user=kravitsjacob@gmail.com
#SBATCH --mail-type=END

module purge
source /curc/sw/anaconda3/2019.07/bin/activate
conda activate water-OPF
python -u main.py '/scratch/summit/jakr3868/water-OPF-v2.2/' 24
