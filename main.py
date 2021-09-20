import sys
sys.path.insert(0, 'src')
import os
import src

import pandapower.converter
import pandas as pd


# Paths for checkpoints
pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v1.0'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')
pathto_case = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.p')
pathto_geninfo = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info.csv')
pathto_geninfo_match = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info_match.csv')
pathto_h5 = os.path.join(pathto_data, 'temp', 'processing_data.h5')

# Paths for external Inputs
pathto_EIA = 'G:\My Drive\Documents (Stored)\data_sets\EIA_theremoelectric_water_use'

# Paths for figures
pathto_figures = os.path.join(pathto_data, 'figures')

def main():

    # Converting matpower
    try:
        pandapower.from_pickle(pathto_case)  # Load checkpoint
    except:
        net = pandapower.converter.from_mpc(pathto_matpowercase)
        print('Success: convert_matpower')
        pandapower.to_pickle(net, pathto_case)  # Save checkpoint
    else:
        net = pandapower.from_pickle(pathto_case)  # Load checkpoint

    # Manual generator matching
    try:
        pd.read_hdf(pathto_h5, 'df_gen_info_match')  # Load checkpoint
    except:
        df_gen_info = pd.read_csv(pathto_geninfo)
        df_gen_info_match = src.generator_match(df_gen_info)
        print('Success: generator_match')
        df_gen_info_match.to_hdf(pathto_h5, key='df_gen_info_match', mode='a')  # Save checkpoint
    else:
        df_gen_info_match = pd.read_hdf(pathto_h5, 'df_gen_info_match')  # Load checkpoint

    # Import EIA data
    try:
        pd.read_hdf(pathto_h5, 'df_EIA')  # Load checkpoint
    except:
        df_EIA = src.import_EIA(pathto_EIA)
        print('Success: import_EIA')
        df_EIA.to_hdf(pathto_h5, key='df_EIA', mode='a')  # Save checkpoint
    else:
        df_EIA = pd.read_hdf(pathto_h5, 'df_EIA')  # Load checkpoint

    # Cooling system information
    try:
        pd.read_hdf(pathto_h5, 'df_gen_info_match_water')  # Load checkpoint
    except:
        df_gen_info_match_water, df_hnwc, fig_regionDistribututionPlotter, fig_regionBoxPlotter, fig_coalPlotter,\
        fig_hnwc_plotter = src.cooling_system_information(df_gen_info_match, df_EIA)
        print('Success: cooling_system_information')
        fig_regionDistribututionPlotter.savefig(os.path.join(pathto_figures, 'uniform water coefficient distribution.pdf'))
        fig_regionBoxPlotter.savefig(os.path.join(pathto_figures, 'region water boxplots.pdf'))
        fig_coalPlotter.savefig(os.path.join(pathto_figures, 'coal scatter kmeans.pdf'))
        fig_hnwc_plotter.savefig(os.path.join(pathto_figures, 'historic nonuniform water coefficient histograms.pdf'))
        df_gen_info_match_water.to_hdf(pathto_h5, key='df_gen_info_match_water', mode='a')  # Save checkpoint
        df_hnwc.to_hdf(pathto_h5, key='df_hnwc', mode='a')  # Save checkpoint
    else:
        df_gen_info_match_water = pd.read_hdf(pathto_h5, 'df_gen_info_match_water')  # Load checkpoint
        df_hnwc = pd.read_hdf(pathto_h5, 'df_hnwc')  # Load checkpoint

    return 0


if __name__ == '__main__':
    main()

