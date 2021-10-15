# water-OPF

This repository contains the analysis and visualization for our water-informed OPF paper. 

## Files
```
.
│   .gitignore
│   LICENSE
│   README.md
│   main.py: Python script to run contents of `analysis.py` and `viz.py`
│   run.sh: Bash script to run `main.py`
│   slurm_nonuniform_run.sh: Slurm batch file to run nonuniform sensitivity analysis
│   slurm_uniform_run.sh: Slurm batch file to run uniform sensitivity analysis
│   water-OPF.yml: Conda environment
│   water-OPF-v2.2 - Shortcut.lnk: Link to local copy of io folder
│   config.ini: Configuration file needed to run `main.py`
│
├───QGIS Manual Spatial Analysis and Cartography
│       EIA_PowerPlants_Locations - Shortcut.lnk
│       Generator Locations.qgz: QGIS project for manual matching synthetic and EIA generators
│       Illinois Synthetic Grid - Shortcut.lnk
│       Illinois Synthetic Grid Gens - Shortcut.lnk
│       North American Rivers and Lakes - Shortcut.lnk
│       UnitedStates_borders - Shortcut.lnk
│
├───src: Source code directory
│   │   to_MATPOWER.m: MATLAB script to convert matpower format
│   │   analysis.py: Functions used for analysis 
│   │   viz.py: Functions used to visualization
├───utils
│       parallel.py: Utility for creating static parallel coordinates 
```

## Notes about getting consistent results
1. Clone this repository to your machine
2. Change to this working directory $`cd water-OPF`
3. Ensure conda is install by running $`conda --version`
    1. If conda is not installed follow instructions [here](https://docs.conda.io/en/latest/miniconda.html).
4. Install the conda environment  $`conda env create -f water-OPF.yml`
5. Run demonstration file $`bash run.sh`
