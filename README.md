# water-OPF

This repository contains the analysis and visualization for our water-informed OPF paper. More information about this project can be found [here](https://osf.io/8h6pc/). That url also provides all the pre-computed input and output files needed for this repository (see section II below).

# I. Contents
```
water-OPF
│   .gitignore
│   config.ini: Configuration file needed to run `main.py`
│   LICENSE
│   main.py: Python script to run contents of `analysis.py` and `viz.py`
│   README.md
│   run.sh: Bash script to run `main.py`
│   slurm_nonuniform_run.sh: Slurm batch file to run nonuniform sensitivity analysis
│   slurm_uniform_run.sh: Slurm batch file to run uniform sensitivity analysis
│   water-OPF.yml: Conda environment
│
├───QGIS Manual Spatial Analysis and Cartography
│       Generator Locations.qgz: QGIS project for manual matching synthetic and EIA generators
│
├───src: Source code directory
│       analysis.py: Functions used for analysis
│       to_MATPOWER.m: MATLAB script to convert matpower format
│       viz.py: Functions used to visualization
│
└───water-OPF-io-v2.3
    ├───figures: Figures that appear in the paper
    │       uniform_water_coefficient_distribution.pdf
    │       region_water_boxplots.pdf
    │       coal_scatter_kmeans.pdf
    │       hnwc_histograms.pdf
    │       historic_load_hist.pdf
    │       effect_of_withdrawal_weight_on_withdrawal.pdf
    │       effect_of_withdrawal_weight_plant_output.pdf
    │       effect_of_withdrawal_weight_line_flows.pdf
    │       decision_tree_total_cost
    │       decision_tree_total_cost.svg
    │       decision_tree_generator_cost
    │       decision_tree_generator_cost.svg
    │       decision_tree_withdrawal
    │       decision_tree_withdrawal.svg
    │       decision_tree_consumption
    │       decision_tree_consumption.svg
    │       nonuniform_sobol_heatmap.pdf
    │
    ├───manual_files: Files generated via manual processes
    │       gen_matches.csv: Generated matching
    │       operational_scenarios.csv: Operational scenarios for nonuniform SA
    │
    ├───tables: Tables that appear in the paper
    │       line_flows.csv
    │       system_information.csv
    │
    ├───generated_files: generated files with 'checkpoint' files useful for debugging
    │   │   case.p
    │   │   case_match.p
    │   │   case_match_water.p
    │   │   hnwc.csv
    │   │   case_match_water_optimize_ready.p
    │   │   nonuniform_sa_results.csv
    │   │   nonuniform_sa_sobol.csv
    │   │   uniform_sa_results.csv
    │   │
    │   └───synthetic_grid
    │           case.mat
    │           gen_info.csv
    │
    ├───figures_manual: Figures manually created. 
    │       Synthetic Generator Map.pdf
    │       decision_tree_total_cost.pdf
    │       decision_tree_consumption.pdf
    │       decision_tree_generator_cost.pdf
    │       decision_tree_withdrawal.pdf
    │
    └───external_data: External data sources with sources
        ├───EIA_theremoelectric_water_use
        │       cooling_detail_2018.xlsx
        │       cooling_detail_2017.xlsx
        │       cooling_detail_2016.xlsx
        │       cooling_detail_2015.xlsx
        │       cooling_detail_2019.xlsx
        │       README.txt
        │       cooling_detail_2014.xlsx
        │
        ├───load_exogenous_parameter
        │       20180101-20200101 MISO Forecasted Cleared & Actual Load.csv
        │       README.txt
        │
        ├───EIA_PowerPlants_Locations
        │       PowerPlants_US_202004.prj
        │       PowerPlants_US_202004.dbf
        │       PowerPlants_US_202004.shx
        │       PowerPlants_US_202004.shp
        │       PowerPlants_US_202004.cpg
        │       PowerPlants_US_202004.sbn
        │       PowerPlants_US_202004.sbx
        │       PowerPlants_US_202004.shp.xml
        │       README.txt
        │
        ├───Illinois Synthetic Grid Gens
        │       gens.shp
        │       gens.dbf
        │       gens.shx
        │       Source.txt
        │
        ├───North American Rivers and Lakes Illinois 2019 Flows
        │       Lakes_and_Rivers_Shapefile_NA_Lakes_and_Rivers_data_hydrography_l_rivers_v2.dbf
        │       Lakes_and_Rivers_Shapefile_NA_Lakes_and_Rivers_data_hydrography_l_rivers_v2.prj
        │       Lakes_and_Rivers_Shapefile_NA_Lakes_and_Rivers_data_hydrography_l_rivers_v2.sbx
        │       Lakes_and_Rivers_Shapefile_NA_Lakes_and_Rivers_data_hydrography_l_rivers_v2.shp
        │       Lakes_and_Rivers_Shapefile_NA_Lakes_and_Rivers_data_hydrography_l_rivers_v2.shp.xml
        │       Lakes_and_Rivers_Shapefile_NA_Lakes_and_Rivers_data_hydrography_l_rivers_v2.shx
        │       Source.txt
        │
        ├───UnitedStates_borders
        │       cb_2018_us_state_20m.cpg
        │       cb_2018_us_state_20m.shp.ea.iso.xml
        │       cb_2018_us_state_20m.shp
        │       cb_2018_us_state_20m.shp.iso.xml
        │       cb_2018_us_state_20m.prj
        │       cb_2018_us_state_20m.dbf
        │       cb_2018_us_state_20m.shx
        │       README.txt
        │
        └───Illinois Synthetic Grid
            │   Source.txt
            │
            └───ACTIVSg200
                    ACTIVSg200.aux
                    ACTIVSg200.EPC
                    ACTIVSg200.pwb
                    ACTIVSg200.pwd
                    ACTIVSg200.RAW
                    ACTIVSg200.tsb
                    ACTIVSg2000.pwd
                    ACTIVSg200_dynamics.dyd
                    ACTIVSg200_dynamics.dyr
                    ACTIVSg200_GIC_data.gic
                    case_ACTIVSg200.m
                    contab_ACTIVSg200.m
                    scenarios_ACTIVSg200.m
```

# II. How to Run
This tutorial assumes the use of [gitbash](https://git-scm.com/downloads) or a Unix-like terminal with github command line usage.
1. This project utilizes conda to manage environments and ensure consistent results. Download [miniconda](https://docs.conda.io/en/latest/miniconda.html) and ensure you can activate it from your terminal by running `$conda activate` 
    * Depending on system configuration, this can be an involved process [here](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473) is a recommended thread.
3. Clone the repository using `$git clone https://github.com/kravitsjacob/water-OPF.git`
4. Download the associated input/output `water-OPF-io-v2.3` data [here](https://osf.io/8h6pc/). Place it in the cloned directory. The tree should appear EXACTLY as it does above. 
5. Change to the current working directory using `$cd <insert_path>/water-OPF`
6. Run the analysis by running `$bash run.sh`
    * To keep this project open source, by default this script does not call the simple Matlab converter functions, and the pre-computed outputs are supplied. However, for the sake of transparency, I have included the Matlab code.

# III. Know Issues
The decision tree plotting package `dtreeviz` and the power system plotting package `pandapower.plotting` have some dependency issues that prevents `dtreeviz` from being used if `pandapower.plotting` is imported. These errors are caughts by `main.py` to avoid a failing exit code, but the decision trees will not be created. If you want to recreate the decision trees, you must comment line 4 in `src/viz.py`. Please reach out if you know a workaround!
