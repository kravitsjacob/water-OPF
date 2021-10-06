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
    pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v2.1'
    n_tasks = os.cpu_count()


# Paths for checkpoints
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')
pathto_geninfo = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info.csv')
pathto_case = os.path.join(pathto_data, 'temp', 'case.p')
pathto_case_match = os.path.join(pathto_data, 'temp', 'case_match.p')
pathto_case_match_water = os.path.join(pathto_data, 'temp', 'case_match_water.p')
pathto_case_match_water_optimize = os.path.join(pathto_data, 'temp', 'case_match_water_optimize_ready.p')

pathto_EIA = os.path.join(pathto_data, 'temp', 'EIA.h5')
pathto_hnwc = os.path.join(pathto_data, 'temp', 'hnwc.csv')
pathto_uniform_sa = os.path.join(pathto_data, 'temp', 'uniform_sa_results.csv')
pathto_nonuniform_sa = os.path.join(pathto_data, 'temp', 'nonuniform_sa_results.csv')
pathto_nonuniform_sa_sobol = os.path.join(pathto_data, 'temp', 'nonuniform_sa_sobol.csv')

# Paths for manual_files
pathto_gen_matches = os.path.join(pathto_data, 'manual_files', 'gen_matches.csv')
pathto_operational_scenarios = os.path.join(pathto_data, 'manual_files', 'operational_scenarios.csv')

# Paths for external Inputs
pathto_EIA_raw = 'G:\My Drive\Documents (Stored)\data_sets\EIA_theremoelectric_water_use'
pathto_load = os.path.join('G:\My Drive\Documents (Stored)\data_sets\load exogenous parameter testing V1 io',
                           '20180101-20200101 MISO Forecasted Cleared & Actual Load.csv')
pathto_gen_locations = 'G:\My Drive\Documents (Stored)\data_sets\Illinois Synthetic Grid Gens\gens.shp'

# Paths for figures
pathto_figures = os.path.join(pathto_data, 'figures')
pathto_tables = os.path.join(pathto_data, 'tables')


def main():
    # Local vars
    n_uniform_steps = 10
    n_nonuniform_samples = 1024 * (2 * 10 + 2)  # for saltelli sampling 1024

    # Setting up grid
    if not os.path.exists(pathto_case):
        net = pandapower.converter.from_mpc(pathto_matpowercase)
        df_gen_info = pd.read_csv(pathto_geninfo)
        net = src.grid_setup(net, df_gen_info)
        print('Success: grid_setup')
        pandapower.to_pickle(net, pathto_case)  # Save checkpoint

    # Manual generator matching
    if not os.path.exists(pathto_case_match):
        net = pandapower.from_pickle(pathto_case)  # Load previous checkpoint
        df_gen_matches = pd.read_csv(pathto_gen_matches)
        net = src.generator_match(net, df_gen_matches)
        print('Success: generator_match')
        pandapower.to_pickle(net, pathto_case_match)  # Save checkpoint

    # Cooling system information
    if not os.path.exists(pathto_case_match_water):
        # Import EIA data
        if os.path.exists(pathto_EIA):
            df_EIA = pd.read_hdf(pathto_EIA, 'df_EIA')  # Load checkpoint
        else:
            df_EIA = src.import_EIA(pathto_EIA_raw)
            print('Success: import_EIA')
            df_EIA.to_hdf(pathto_EIA, key='df_EIA', mode='w')  # Save checkpoint

        net = pandapower.from_pickle(pathto_case_match)  # Load previous checkpoint
        net, df_hnwc, fig_region_distributution_plotter, fig_region_box_plotter, fig_coal_plotter, fig_hnwc_plotter = \
            src.cooling_system_information(net, df_EIA)
        print('Success: cooling_system_information')
        fig_region_distributution_plotter.savefig(os.path.join(pathto_figures, 'uniform water coefficient distribution.pdf'))
        fig_region_box_plotter.savefig(os.path.join(pathto_figures, 'region water boxplots.pdf'))
        fig_coal_plotter.savefig(os.path.join(pathto_figures, 'coal scatter kmeans.pdf'))
        fig_hnwc_plotter.savefig(os.path.join(pathto_figures, 'historic nonuniform water coefficient histograms.pdf'))
        pandapower.to_pickle(net, pathto_case_match_water)  # Save checkpoint
        df_hnwc.to_csv(pathto_hnwc, index=False)  # Save checkpoint

    # Prepare network for optimization
    if not os.path.exists(pathto_case_match_water_optimize):
        net = pandapower.from_pickle(pathto_case_match_water)  # Load previous checkpoint
        net = src.optimization_information(net)
        pandapower.to_pickle(net, pathto_case_match_water_optimize)

    # Uniform SA
    if not os.path.exists(pathto_uniform_sa):
        net = pandapower.from_pickle(pathto_case_match_water_optimize)  # Load previous checkpoint
        df_uniform = src.uniform_sa(net, n_tasks, n_uniform_steps)
        df_uniform.to_csv(pathto_uniform_sa, index=False)  # Save checkpoint

    # Uniform SA Data Viz
    if not os.path.exists(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Line Flows.pdf')):
        net = pandapower.from_pickle(pathto_case_match_water_optimize)  # Load previous checkpoint
        df_uniform = pd.read_csv(pathto_uniform_sa)  # Load previous checkpoint
        fig_a, fig_b, df_line_flows, fig_c = src.uniform_sa_dataviz(df_uniform, net)
        fig_a.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Withdrawal.pdf'))
        fig_b.fig.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Plant Output.pdf'))
        df_line_flows.to_csv(os.path.join(pathto_tables, 'line_flows.csv'), index=False)
        fig_c.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Line Flows.pdf'))

    # Uniform SA Trees
    if not os.path.exists(os.path.join(pathto_figures, 'Total Cost (Dollar) Tree.pdf')):
        net = pandapower.from_pickle(pathto_case_match_water_optimize)  # Load previous checkpoint
        df_uniform = pd.read_csv(pathto_uniform_sa)  # Load previous checkpoint
        drawing_ls = src.uniform_sa_tree(df_uniform, net)
        renderPDF.drawToFile(drawing_ls[0], os.path.join(pathto_figures, 'Total Cost (Dollar) Tree.pdf'))
        renderPDF.drawToFile(drawing_ls[1], os.path.join(pathto_figures, 'Generator Cost (Dollar) Tree.pdf'))
        renderPDF.drawToFile(drawing_ls[2], os.path.join(pathto_figures, 'Water Withdrawal (Gallon) Tree.pdf'))
        renderPDF.drawToFile(drawing_ls[3], os.path.join(pathto_figures, 'Water Consumption (Gallon) Tree.pdf'))

    # Nonuniform SA
    if not os.path.exists(pathto_nonuniform_sa_sobol):
        net = pandapower.from_pickle(pathto_case_match_water_optimize)  # Load previous checkpoint
        df_hnwc = pd.read_csv(pathto_hnwc)  # Load previous checkpoint
        df_operation = pd.read_csv(pathto_operational_scenarios)
        df_nonuniform, df_nonuniform_sobol = src.nonuniform_sa(
            df_hnwc, df_operation, n_tasks, n_nonuniform_samples, net
        )
        df_nonuniform.to_csv(pathto_nonuniform_sa, index=False)  # Save checkpoint
        df_nonuniform_sobol.to_csv(pathto_nonuniform_sa_sobol, index=False)  # Save checkpoint

    # Sobol Visualization
    if not os.path.exists(os.path.join(pathto_figures, 'First Order Heatmap.pdf')):
        net = pandapower.from_pickle(pathto_case_match_water_optimize)  # Load previous checkpoint
        df_nonuniform_sobol = pd.read_csv(pathto_nonuniform_sa_sobol)
        nonuniform_sobol_fig = src.nonuniform_sobol_viz(df_nonuniform_sobol, net)
        nonuniform_sobol_fig.fig.savefig(os.path.join(pathto_figures, 'First Order Heatmap.pdf'))

    # Historic Load Generation
    if not os.path.exists(os.path.join(pathto_figures, 'Load Distribution.pdf')):
        df_historic_loads = pd.read_csv(pathto_load)
        src.historic_load_viz(df_historic_loads).savefig(os.path.join(pathto_figures, 'Load Distribution.pdf'))

    # System Information Table
    if not os.path.exists(os.path.join(pathto_tables, 'system_information.csv')):
        net = pandapower.from_pickle(pathto_case_match_water_optimize)  # Load previous checkpoint
        df_system = src.get_system_information(net)
        df_system.to_csv(os.path.join(pathto_tables, 'system_information.csv'), index=False)

    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
