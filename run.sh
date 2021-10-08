#!/bin/bash

conda activate water-OPF

# Input Files
pathto_case='G:\\My Drive\\Documents (Stored)\\data_sets\Illinois Synthetic Grid\ACTIVSg200\\case_ACTIVSg200.m'

# Temp Files
#pathto_case_info='G:\My Drive\Documents (Stored)\data_sets\water-OPF-v2.1\temp\gen_info.csv',
#pathto_case_export='G:\My Drive\Documents (Stored)\data_sets\water-OPF-v2.1\temp\case.mat'

# Convert Matlab format
cd src
matlab -nojvm -r 'to_MATPOWER; exit;'
cd ..

python main.py
