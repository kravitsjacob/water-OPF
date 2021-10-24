#!/bin/bash

# Load conda commands
eval "$(conda shell.bash hook)"

# Create conda environment
conda env create -f water-OPF.yml

# Load conda environment
conda activate water-OPF

# Convert Matlab format (output files supplied)
#cd src
#matlab -nojvm -r 'to_MATPOWER; exit;'
#cd ..

# Run analysis
python main.py -c config.ini