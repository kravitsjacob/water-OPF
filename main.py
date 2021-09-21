import sys
sys.path.insert(0, 'src')
import os
import src

import pandapower.converter
import pandas as pd


# Paths for checkpoints
pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v1.0'
pathto_case = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.p')
pathto_geninfo = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info.csv')
pathto_geninfo_match = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info_match.csv')
pathto_h5 = os.path.join(pathto_data, 'temp', 'processing_data.h5')

# Paths for external Inputs
pathto_EIA = 'G:\My Drive\Documents (Stored)\data_sets\EIA_theremoelectric_water_use'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')

# Paths for figures
pathto_figures = os.path.join(pathto_data, 'figures')

def main():

    # Converting matpower
    if os.path.exists(pathto_case):
        net = pandapower.from_pickle(pathto_case)  # Load checkpoint
    else:
        net = pandapower.converter.from_mpc(pathto_matpowercase)
        print('Success: convert_matpower')
        pandapower.to_pickle(net, pathto_case)  # Save checkpoint

    # Manual generator matching
    s = pd.HDFStore(pathto_h5)
    if 'df_gen_info_match' in s:
        s.close()
        df_gen_info_match = pd.read_hdf(pathto_h5, 'df_gen_info_match')  # Load checkpoint
    else:
        df_gen_info = pd.read_csv(pathto_geninfo)
        df_gen_info_match = src.generator_match(df_gen_info)
        print('Success: generator_match')
        df_gen_info_match.to_hdf(pathto_h5, key='df_gen_info_match', mode='a')  # Save checkpoint

    # Import EIA data
    s = pd.HDFStore(pathto_h5)
    if 'df_EIA' in s:
        s.close()
        df_EIA = pd.read_hdf(pathto_h5, 'df_EIA')  # Load checkpoint
    else:
        df_EIA = src.import_EIA(pathto_EIA)
        print('Success: import_EIA')
        df_EIA.to_hdf(pathto_h5, key='df_EIA', mode='a')  # Save checkpoint

    # Cooling system information
    s = pd.HDFStore(pathto_h5)
    if 'df_gen_info_match_water' in s:
        s.close()
        df_gen_info_match_water = pd.read_hdf(pathto_h5, 'df_gen_info_match_water')  # Load checkpoint
        df_hnwc = pd.read_hdf(pathto_h5, 'df_hnwc')  # Load checkpoint
    else:
        df_gen_info_match_water, df_hnwc, fig_regionDistribututionPlotter, fig_regionBoxPlotter, fig_coalPlotter,\
        fig_hnwc_plotter = src.cooling_system_information(df_gen_info_match, df_EIA)
        print('Success: cooling_system_information')
        fig_regionDistribututionPlotter.savefig(os.path.join(pathto_figures, 'uniform water coefficient distribution.pdf'))
        fig_regionBoxPlotter.savefig(os.path.join(pathto_figures, 'region water boxplots.pdf'))
        fig_coalPlotter.savefig(os.path.join(pathto_figures, 'coal scatter kmeans.pdf'))
        fig_hnwc_plotter.savefig(os.path.join(pathto_figures, 'historic nonuniform water coefficient histograms.pdf'))
        df_gen_info_match_water.to_hdf(pathto_h5, key='df_gen_info_match_water', mode='a')  # Save checkpoint
        df_hnwc.to_hdf(pathto_h5, key='df_hnwc', mode='a')  # Save checkpoint

    return 0


if __name__ == '__main__':
    main()

