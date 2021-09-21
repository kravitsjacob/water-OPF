import sys
sys.path.insert(0, 'src')
import os
import src
import multiprocessing

import pandapower.converter
import pandas as pd
from reportlab.graphics import renderPDF


if len(sys.argv) > 1:
    pathto_data = sys.argv[1]
    n_tasks = int(sys.argv[2])
else:
    pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v1.0'
    n_tasks = os.cpu_count()


# Paths for checkpoints
pathto_case = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.p')
pathto_gen_info_match = os.path.join(pathto_data, 'temp', 'gen_info_match.csv')
pathto_gen_info_match_water = os.path.join(pathto_data, 'temp', 'gen_info_match_water.csv')
pathto_EIA = os.path.join(pathto_data, 'temp', 'EIA.h5')
pathto_hnwc = os.path.join(pathto_data, 'temp', 'hnwc.csv')
pathto_uniform_sa = os.path.join(pathto_data, 'temp', 'uniform_sa_results.csv')

# Paths for external Inputs
pathto_EIA_raw = 'G:\My Drive\Documents (Stored)\data_sets\EIA_theremoelectric_water_use'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')
pathto_geninfo = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info.csv')

# Paths for figures
pathto_figures = os.path.join(pathto_data, 'figures')

def main():
    # Initialize vars
    uniform_factor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)',
                           'Uniform Loading Coefficient', 'Uniform Water Coefficient']
    obj_labs = ['Total Cost ($)', 'Generator Cost ($)',	'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)']

    # Setting up grid
    if os.path.exists(pathto_case):
        net = pandapower.from_pickle(pathto_case)  # Load checkpoint
    else:
        net = pandapower.converter.from_mpc(pathto_matpowercase)
        net = src.grid_setup(net)
        print('Success: convert_matpower')
        pandapower.to_pickle(net, pathto_case)  # Save checkpoint

    # Manual generator matching
    if os.path.exists(pathto_gen_info_match):
        df_gen_info_match = pd.read_csv(pathto_gen_info_match)  # Load checkpoint
    else:
        df_gen_info = pd.read_csv(pathto_geninfo)
        df_gen_info_match = src.generator_match(df_gen_info)
        print('Success: generator_match')
        df_gen_info_match.to_csv(pathto_gen_info_match, index=False)  # Save checkpoint

    # Cooling system information
    if os.path.exists(pathto_gen_info_match_water):
        df_gen_info_match_water = pd.read_csv(pathto_gen_info_match_water)  # Load checkpoint
        df_hnwc = pd.read_csv(pathto_hnwc)  # Load checkpoint
    else:
        # Import EIA data
        if os.path.exists(pathto_EIA):
            df_EIA = pd.read_hdf(pathto_EIA, 'df_EIA')  # Load checkpoint
        else:
            df_EIA = src.import_EIA(pathto_EIA_raw)
            print('Success: import_EIA')
            df_EIA.to_hdf(pathto_EIA, key='df_EIA', mode='w')  # Save checkpoint
        df_gen_info_match_water, df_hnwc, fig_regionDistribututionPlotter, fig_regionBoxPlotter, fig_coalPlotter,\
        fig_hnwc_plotter = src.cooling_system_information(df_gen_info_match, df_EIA)
        print('Success: cooling_system_information')
        fig_regionDistribututionPlotter.savefig(os.path.join(pathto_figures, 'uniform water coefficient distribution.pdf'))
        fig_regionBoxPlotter.savefig(os.path.join(pathto_figures, 'region water boxplots.pdf'))
        fig_coalPlotter.savefig(os.path.join(pathto_figures, 'coal scatter kmeans.pdf'))
        fig_hnwc_plotter.savefig(os.path.join(pathto_figures, 'historic nonuniform water coefficient histograms.pdf'))
        df_gen_info_match_water.to_csv(pathto_gen_info_match_water, index=False)  # Save checkpoint
        df_hnwc.to_csv(pathto_hnwc, index=False)  # Save checkpoint

    # Uniform SA
    if os.path.exists(pathto_uniform_sa):
        df_uniform = pd.read_csv(pathto_uniform_sa)  # Load Checkpoint
    else:
        df_uniform = src.uniform_sa(df_gen_info_match_water, net, n_tasks, 10, uniform_factor_labs, obj_labs)
        df_uniform.to_csv(pathto_uniform_sa, index=False)

    # Enter data viz here

    # Uniform SA Trees
    if not os.path.exists(os.path.join(pathto_figures, 'Total Cost (Dollar) Tree.pdf')):
        drawing_ls = src.uniform_sa_tree(df_uniform, obj_labs, uniform_factor_labs)
        renderPDF.drawToFile(drawing_ls[0], os.path.join(pathto_figures, 'Total Cost (Dollar) Tree.pdf'))
        renderPDF.drawToFile(drawing_ls[1], os.path.join(pathto_figures, 'Generator Cost (Dollar) Tree.pdf'))
        renderPDF.drawToFile(drawing_ls[2], os.path.join(pathto_figures, 'Water Withdrawal (Gallon) Tree.pdf'))
        renderPDF.drawToFile(drawing_ls[3], os.path.join(pathto_figures, 'Water Consumption (Gallon) Tree.pdf'))
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
