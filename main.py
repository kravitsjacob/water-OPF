""" Python script to run analysis """

import os
import multiprocessing
import configparser
import argparse

import pandapower.converter
import pandas as pd

from src import analysis
from src import viz


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
    path_to_eia_raw = os.path.join(path_to_data, config_inputs['EXTERNAL INPUTS']['EIA_raw'])
    path_to_load = os.path.join(path_to_data, config_inputs['EXTERNAL INPUTS']['load'])

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

    # Parallel information
    n_tasks = os.cpu_count()
    if argparse_inputs.n_tasks:  # Command line gets priority
        n_tasks = argparse_inputs.n_tasks
        print(f'Command line inputs get priority over config file {n_tasks=}')

    # Store inputs
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

    # Inputs
    inputs = input_parse()

    # Setting up grid
    if not os.path.exists(inputs['path_to_case']):
        net = pandapower.converter.from_mpc(inputs['path_to_matpowercase'])
        df_gen_info = pd.read_csv(inputs['path_to_geninfo'])
        net = analysis.grid_setup(net, df_gen_info)
        print('Success: grid_setup')
        pandapower.to_pickle(net, inputs['path_to_case'])  # Save checkpoint

    # Manual generator matching
    if not os.path.exists(inputs['path_to_case_match']):
        net = pandapower.from_pickle(inputs['path_to_case'])  # Load previous checkpoint
        df_gen_matches = pd.read_csv(inputs['path_to_gen_matches'])
        net = analysis.generator_match(net, df_gen_matches)
        print('Success: generator_match')
        pandapower.to_pickle(net, inputs['path_to_case_match'])  # Save checkpoint

    # Cooling system information
    if not os.path.exists(inputs['path_to_case_match_water']):
        # Import EIA data
        if os.path.exists(inputs['path_to_eia']):
            df_eia = pd.read_hdf(inputs['path_to_eia'], 'df_eia')  # Load checkpoint
        else:
            df_eia = analysis.import_eia(inputs['path_to_eia_raw'])
            print('Success: import_eia')
            df_eia.to_hdf(inputs['path_to_eia'], key='df_eia', mode='w')  # Save checkpoint

        net = pandapower.from_pickle(inputs['path_to_case_match'])  # Load previous checkpoint
        net, df_hnwc, df_region, df_gen_info = analysis.cooling_system_information(net, df_eia)
        print('Success: cooling_system_information')
        pandapower.to_pickle(net, inputs['path_to_case_match_water'])  # Save checkpoint
        df_hnwc.to_csv(inputs['path_to_hnwc'], index=False)  # Save checkpoint

        # Plotting
        viz.uniform_water_coefficient_distribution(df_region).figure.savefig(
            os.path.join(inputs['path_to_figures'], 'uniform_water_coefficient_distribution.pdf')
        )
        viz.region_water_boxplots(df_region).fig.savefig(
            os.path.join(inputs['path_to_figures'], 'region_water_boxplots.pdf')
        )
        viz.coal_scatter_kmeans(df_region).fig.savefig(
            os.path.join(inputs['path_to_figures'], 'coal_scatter_kmeans.pdf')
        )
        viz.hnwc_histograms(df_hnwc, df_gen_info).fig.savefig(
            os.path.join(inputs['path_to_figures'], 'hnwc_histograms.pdf')
        )

    # Prepare network for optimization
    if not os.path.exists(inputs['path_to_case_match_water_optimize']):
        net = pandapower.from_pickle(inputs['path_to_case_match_water'])  # Load previous checkpoint
        net = analysis.optimization_information(net)
        pandapower.to_pickle(net, inputs['path_to_case_match_water_optimize'])

    # Uniform SA
    if not os.path.exists(inputs['path_to_uniform_sa']):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_uniform = analysis.uniform_sa(net, inputs['n_tasks'], n_uniform_steps)
        df_uniform.to_csv(inputs['path_to_uniform_sa'], index=False)  # Save checkpoint

    # Uniform SA data visualization
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'effect_of_withdrawal_weight_line_flows.pdf')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_uniform = pd.read_csv(inputs['path_to_uniform_sa'])  # Load previous checkpoint
        # Convert to generator information dataframe
        df_gen_info = analysis.network_to_gen_info(net)

        # Generator visualization
        viz.effect_of_withdrawal_weight_on_withdrawal(
            df_uniform, net.uniform_input_factor_labs, net.objective_labs
        ).savefig(os.path.join(inputs['path_to_figures'], 'effect_of_withdrawal_weight_on_withdrawal.pdf'))
        df_plant_capacity_ratio = analysis.get_plant_output_ratio(
            df_uniform, df_gen_info, net.uniform_input_factor_labs, net.objective_labs
        )
        viz.effect_of_withdrawal_weight_plant_output(df_plant_capacity_ratio).fig.savefig(
            os.path.join(inputs['path_to_figures'], 'effect_of_withdrawal_weight_plant_output.pdf')
        )

        # Line flow visualization
        net_diff, df_line_flows = analysis.get_line_flow_difference(net)
        df_line_flows.to_csv(os.path.join(inputs['path_to_tables'], 'line_flows.csv'), index=False)
        viz.effect_of_withdrawal_weight_line_flows(net_diff).savefig(
            os.path.join(inputs['path_to_figures'], 'effect_of_withdrawal_weight_line_flows.pdf')
        )

    # Uniform SA trees
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'decision_tree_total_cost.svg')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_uniform = pd.read_csv(inputs['path_to_uniform_sa'])  # Load previous checkpoint

        # Fit trees
        mods = analysis.fit_single_models(df_uniform, net.objective_labs, net.uniform_input_factor_labs)

        # Visualize trees
        try:
            viz.decision_tree(mods, 'Total Cost ($)', df_uniform, net.uniform_input_factor_labs).save(
                os.path.join(inputs['path_to_figures'], 'decision_tree_total_cost.svg')
            )
            viz.decision_tree(mods, 'Generator Cost ($)', df_uniform, net.uniform_input_factor_labs).save(
                os.path.join(inputs['path_to_figures'], 'decision_tree_generator_cost.svg')
            )
            viz.decision_tree(mods, 'Water Withdrawal (Gallon)', df_uniform, net.uniform_input_factor_labs).save(
                os.path.join(inputs['path_to_figures'], 'decision_tree_withdrawal.svg')
            )
            viz.decision_tree(mods, 'Water Consumption (Gallon)', df_uniform, net.uniform_input_factor_labs).save(
                os.path.join(inputs['path_to_figures'], 'decision_tree_consumption.svg')
            )
        except AttributeError:
            print('AttributeError: Cannot use both `dtreeviz` and `pandapower.plotting`')

    # Nonuniform SA
    if not os.path.exists(inputs['path_to_nonuniform_sa_sobol']):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_hnwc = pd.read_csv(inputs['path_to_hnwc'])  # Load previous checkpoint
        df_operation = pd.read_csv(inputs['path_to_operational_scenarios'])
        df_nonuniform, df_nonuniform_sobol = analysis.nonuniform_sa(
            df_hnwc, df_operation, inputs['n_tasks'], n_nonuniform_samples, net
        )
        df_nonuniform.to_csv(inputs['path_to_nonuniform_sa'], index=False)  # Save checkpoint
        df_nonuniform_sobol.to_csv(inputs['path_to_nonuniform_sa_sobol'], index=False)  # Save checkpoint

    # Sobol visualization
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'nonuniform_sobol_heatmap.pdf')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_nonuniform_sobol = pd.read_csv(inputs['path_to_nonuniform_sa_sobol'])
        df_gen_info = analysis.network_to_gen_info(net)
        viz.nonuniform_sobol_heatmap(df_nonuniform_sobol, df_gen_info).fig.savefig(
            os.path.join(inputs['path_to_figures'], 'nonuniform_sobol_heatmap.pdf')
        )

    # Historic load generation
    if not os.path.exists(os.path.join(inputs['path_to_figures'], 'historic_load_hist.pdf')):
        df_historic_loads = pd.read_csv(inputs['path_to_load'])
        viz.historic_load_hist(df_historic_loads).savefig(
            os.path.join(inputs['path_to_figures'], 'historic_load_hist.pdf')
        )

    # System information table
    if not os.path.exists(os.path.join(inputs['path_to_tables'], 'system_information.csv')):
        net = pandapower.from_pickle(inputs['path_to_case_match_water_optimize'])  # Load previous checkpoint
        df_gen_info = analysis.network_to_gen_info(net)
        df_system = viz.system_information(df_gen_info)
        df_system.to_csv(os.path.join(inputs['path_to_tables'], 'system_information.csv'), index=False)

    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
