import sys
import os
import multiprocessing
import configparser
import argparse

import pandapower.converter
import pandas as pd
from reportlab.graphics import renderPDF

sys.path.insert(0, 'src')
import src  # noqa: E402


def input_parse():
    # Local vars
    config_inputs = configparser.ConfigParser()
    argparse_inputs = argparse.ArgumentParser()

    # Command line arguments (these get priority)
    argparse_inputs.add_argument(
        '-c',
        '--config_file',
        type=str,
        action='store',
        help='Path to configuration file',
        required=True
    )
    argparse_inputs.add_argument(
        '-n',
        '--n_tasks',
        type=int,
        action='store',
        help='Number of tasks for parallel processing',
        required=False
    )
    argparse_inputs.add_argument(
        '-p',
        '--path_to_data',
        type=str,
        action='store',
        help='Path to main data folder',
        required=False
    )

    # Parse arguments
    argparse_inputs = argparse_inputs.parse_args()

    # Parse config file
    config_inputs.read(argparse_inputs.config_file)

    # Path for Main IO
    path_to_data = config_inputs['MAIN IO']['data']
    if argparse_inputs.path_to_data:  # Command line gets priority
        path_to_data = argparse_inputs.path_to_data
        print(f'Command line inputs get priority over config file {path_to_data=}')

    # Paths for manual_files
    path_to_gen_matches = os.path.join(path_to_data, config_inputs['MANUAL FILES']['gen_matches'])
    path_to_operational_scenarios = os.path.join(path_to_data, config_inputs['MANUAL FILES']['operational_scenarios'])

    # Paths for figures/tables
    path_to_figures = os.path.join(path_to_data, config_inputs['FIGURES']['figures'])
    path_to_tables = os.path.join(path_to_data, config_inputs['FIGURES']['tables'])

    # Paths for external Inputs
    path_to_eia_raw = config_inputs['EXTERNAL INPUTS']['EIA_raw']
    path_to_load = config_inputs['EXTERNAL INPUTS']['load']

    # Paths for checkpoints
    path_to_matpowercase = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['matpowercase'])
    path_to_geninfo = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['geninfo'])
    path_to_case = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['case'])
    path_to_case_match = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['case_match'])
    path_to_case_match_water = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['case_match_water'])
    path_to_case_match_water_optimize = os.path.join(
        path_to_data, config_inputs['CHECKPOINTS']['case_match_water_optimize']
    )
    path_to_eia = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['EIA'])
    path_to_hnwc = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['hnwc'])
    path_to_uniform_sa = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['uniform_sa'])
    path_to_nonuniform_sa = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['nonuniform_sa'])
    path_to_nonuniform_sa_sobol = os.path.join(path_to_data, config_inputs['CHECKPOINTS']['nonuniform_sa_sobol'])

    # Parallel Information
    n_tasks = os.cpu_count()
    if argparse_inputs.n_tasks:  # Command line gets priority
        n_tasks = argparse_inputs.n_tasks
        print(f'Command line inputs get priority over config file {n_tasks=}')

    # Store Inputs
    inputs = {
        'path_to_gen_matches': path_to_gen_matches,
        'path_to_operational_scenarios': path_to_operational_scenarios,
        'path_to_figures': path_to_figures,
        'path_to_tables': path_to_tables,
        'path_to_matpowercase': path_to_matpowercase,
        'path_to_geninfo': path_to_geninfo,
        'path_to_case': path_to_case,
        'path_to_case_match': path_to_case_match,
        'path_to_case_match_water': path_to_case_match_water,
        'path_to_case_match_water_optimize': path_to_case_match_water_optimize,
        'path_to_eia': path_to_eia,
        'path_to_hnwc': path_to_hnwc,
        'path_to_uniform_sa': path_to_uniform_sa,
        'path_to_nonuniform_sa': path_to_nonuniform_sa,
        'path_to_nonuniform_sa_sobol': path_to_nonuniform_sa_sobol,
        'path_to_eia_raw': path_to_eia_raw,
        'path_to_load': path_to_load,
        'n_tasks': n_tasks
    }

    return inputs


def main():
    # Local vars
    n_uniform_steps = 10
    n_nonuniform_samples = 1024 * (2 * 10 + 2)  # for saltelli sampling 1024

    n_uniform_steps = 2
    n_nonuniform_samples = 10 #1024 * (2 * 10 + 2)  # for saltelli sampling 1024

    # Inputs
    inputs = input_parse()

    # Setting up grid
    if not os.path.exists(inputs['path_to_case']):
        net = pandapower.converter.from_mpc(inputs['path_to_matpowercase'])
        df_gen_info = pd.read_csv(inputs['path_to_geninfo'])
        net = src.grid_setup(net, df_gen_info)
        print('Success: grid_setup')
        pandapower.to_pickle(net, inputs['path_to_case'])  # Save checkpoint

    # Manual generator matching
    if not os.path.exists(inputs['path_to_case_match']):
        net = pandapower.from_pickle(inputs['path_to_case'])  # Load previous checkpoint
        df_gen_matches = pd.read_csv(inputs['path_to_gen_matches'])
        net = src.generator_match(net, df_gen_matches)
        print('Success: generator_match')
        pandapower.to_pickle(net, inputs['path_to_case_match'])  # Save checkpoint

    # Cooling system information
    if not os.path.exists(inputs['path_to_case_match_water']):
        # Import EIA data
        if os.path.exists(inputs['path_to_eia']):
            df_eia = pd.read_hdf(inputs['path_to_eia'], 'df_eia')  # Load checkpoint
        else:
            df_eia = src.import_eia(inputs['path_to_eia_raw'])
            print('Success: import_eia')
            df_eia.to_hdf(inputs['path_to_eia'], key='df_eia', mode='w')  # Save checkpoint

        net = pandapower.from_pickle(inputs['path_to_case_match'])  # Load previous checkpoint
        net, df_hnwc, fig_region_distributution_plotter, fig_region_box_plotter, fig_coal_plotter, fig_hnwc_plotter = \
            src.cooling_system_information(net, df_eia)
        print('Success: cooling_system_information')
        fig_region_distributution_plotter.savefig(
            os.path.join(inputs['path_to_figures'], 'uniform water coefficient distribution.pdf')
        )
        fig_region_box_plotter.savefig(os.path.join(inputs['path_to_figures'], 'region water boxplots.pdf'))
        fig_coal_plotter.savefig(os.path.join(inputs['path_to_figures'], 'coal scatter kmeans.pdf'))
        fig_hnwc_plotter.savefig(
            os.path.join(inputs['path_to_figures'], 'historic nonuniform water coefficient histograms.pdf')
        )
        pandapower.to_pickle(net, inputs['path_to_case_match_water'])  # Save checkpoint
        df_hnwc.to_csv(inputs['path_to_hnwc'], index=False)  # Save checkpoint

    # Prepare network for optimization
    if not os.path.exists(inputs['path_to_case_match_water_optimize']):
        net = pandapower.from_pickle(inputs['path_to_case_match_water'])  # Load previous checkpoint
        net = src.optimization_information(net)
        pandapower.to_pickle(net, inputs['path_to_case_match_water_optimize'])

    # Uniform SA
    if not os.path.exists(inputs['path_to_uniform_sa']):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_uniform = src.uniform_sa(net, inputs['n_tasks'], n_uniform_steps)
        df_uniform.to_csv(inputs['path_to_uniform_sa'], index=False)  # Save checkpoint

    # Uniform SA Data Viz
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'Effect of Withdrawal Weight on Line Flows.pdf')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_uniform = pd.read_csv(inputs['path_to_uniform_sa'])  # Load previous checkpoint
        fig_a, fig_b, df_line_flows, fig_c = src.uniform_sa_dataviz(df_uniform, net)
        fig_a.savefig(os.path.join(inputs['path_to_figures'], 'Effect of Withdrawal Weight on Withdrawal.pdf'))
        fig_b.fig.savefig(os.path.join(inputs['path_to_figures'], 'Effect of Withdrawal Weight on Plant Output.pdf'))
        df_line_flows.to_csv(os.path.join(inputs['path_to_tables'], 'line_flows.csv'), index=False)
        fig_c.savefig(os.path.join(inputs['path_to_figures'], 'Effect of Withdrawal Weight on Line Flows.pdf'))

    # Uniform SA Trees
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'Total Cost (Dollar) Tree.pdf')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_uniform = pd.read_csv(inputs['path_to_uniform_sa'])  # Load previous checkpoint
        drawing_ls = src.uniform_sa_tree(df_uniform, net)
        if len(drawing_ls) == 4:
            renderPDF.drawToFile(
                drawing_ls[0], os.path.join(inputs['path_to_figures'], 'Total Cost (Dollar) Tree.pdf')
            )
            renderPDF.drawToFile(
                drawing_ls[1], os.path.join(inputs['path_to_figures'], 'Generator Cost (Dollar) Tree.pdf')
            )
            renderPDF.drawToFile(
                drawing_ls[2], os.path.join(inputs['path_to_figures'], 'Water Withdrawal (Gallon) Tree.pdf')
            )
            renderPDF.drawToFile(
                drawing_ls[3], os.path.join(inputs['path_to_figures'], 'Water Consumption (Gallon) Tree.pdf')
            )

    # Nonuniform SA
    if not os.path.exists(inputs['path_to_nonuniform_sa_sobol']):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_hnwc = pd.read_csv(inputs['path_to_hnwc'])  # Load previous checkpoint
        df_operation = pd.read_csv(inputs['path_to_operational_scenarios'])
        df_nonuniform, df_nonuniform_sobol = src.nonuniform_sa(
            df_hnwc, df_operation, inputs['n_tasks'], n_nonuniform_samples, net
        )
        df_nonuniform.to_csv(inputs['path_to_nonuniform_sa'], index=False)  # Save checkpoint
        df_nonuniform_sobol.to_csv(inputs['path_to_nonuniform_sa_sobol'], index=False)  # Save checkpoint

    # Sobol Visualization
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'First Order Heatmap.pdf')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_nonuniform_sobol = pd.read_csv(inputs['path_to_nonuniform_sa_sobol'])
        nonuniform_sobol_fig = src.nonuniform_sobol_viz(df_nonuniform_sobol, net)
        nonuniform_sobol_fig.fig.savefig(os.path.join(inputs['path_to_figures'], 'First Order Heatmap.pdf'))

    # Historic Load Generation
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'Load Distribution.pdf')):
        df_historic_loads = pd.read_csv(inputs['path_to_load'])
        src.historic_load_viz(df_historic_loads).savefig(
            os.path.join(inputs['path_to_figures'], 'Load Distribution.pdf')
        )

    # System Information Table
    if not os.path.exists(os.path.join(inputs['path_to_tables'], 'system_information.csv')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_system = src.get_system_information(net)
        df_system.to_csv(os.path.join(inputs['path_to_tables'], 'system_information.csv'), index=False)

    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
