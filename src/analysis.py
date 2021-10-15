import os
import itertools
import copy

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import dask.dataframe as dd
import pandapower as pp
from sklearn import tree

def network_to_gen_info(net):
    # Initialize local vars
    gen_types = ['gen', 'sgen', 'ext_grid']
    df_gen_info = pd.DataFrame()

    # Convert generator information dataframe
    for gen_type in gen_types:
        df_gen_info = df_gen_info.append(getattr(net, gen_type))

    df_gen_info = df_gen_info.reset_index(drop=True)  # Important to eliminate duplicated indices

    return df_gen_info


def add_gen_info_to_network(df_gen_info, net):
    # Initialize local vars
    gen_types = ['gen', 'sgen', 'ext_grid']

    # Add information
    for gen_type in gen_types:
        setattr(net, gen_type, getattr(net, gen_type).merge(df_gen_info))

    return net


def grid_setup(net, df_gen_info):

    # Initialize local vars
    gen_types = ['gen', 'sgen', 'ext_grid']

    # Add pandapower index
    for gen_type in gen_types:
        getattr(net, gen_type)['MATPOWER Index'] = getattr(net, gen_type)['bus'] + 1

    # Add generator information
    net = add_gen_info_to_network(df_gen_info, net)

    return net


def generator_match(net, df_gen_matches):
    # Anonymous plant names
    powerworld_plants = df_gen_matches['POWERWORLD Plant Name'].unique()
    anonymous_plants = [f'Plant {i}' for i in range(1, len(powerworld_plants) + 1)]
    d = dict(zip(powerworld_plants, anonymous_plants))
    df_gen_matches['Plant Name'] = df_gen_matches['POWERWORLD Plant Name'].map(d)

    # Add generator information
    net = add_gen_info_to_network(df_gen_matches, net)

    return net


def import_eia(path_to_eia):
    # Local Vars
    years = ['2019', '2018', '2017', '2016', '2015', '2014']
    df_list = []

    # Import all dataframes
    for i in years:
        path = os.path.join(path_to_eia, 'cooling_detail_' + i + '.xlsx')
        print(i)

        # Import Dataframe
        df_temp = pd.read_excel(path, header=2)

        # Replace space values with nan values
        df_temp = df_temp.replace(r'^\s*$', np.nan, regex=True)
        df_list.append(df_temp)

    # Concat Dataframes into Single Dataframe
    df = pd.concat(df_list)
    return df


def get_cooling_system(df_eia, df_gen_info):
    # Matches from manual analysis of EIA dataset
    df_eia = df_eia.drop_duplicates(subset='Plant Name', keep='first')
    df_gen_info['923 Cooling Type'] = df_gen_info.merge(
        df_eia,
        right_on='Plant Name',
        left_on='EIA Plant Name',
        how='left'
    )['923 Cooling Type']

    # Assumptions about cooling systems
    df_gen_info.loc[df_gen_info['MATPOWER Fuel'] == 'wind', '923 Cooling Type'] = \
        'No Cooling System'  # Wind is assumed to not use water
    df_gen_info.loc[(df_gen_info['MATPOWER Type'] == 'GT') & (df_gen_info['MATPOWER Fuel'] == 'ng') &
                    (df_gen_info['MATPOWER Capacity (MW)'] < 30), '923 Cooling Type'] = \
        'No Cooling System'  # Assume Small Capacity Natural Gas Turbines Don't Have Cooling System

    # One off matching based on searching
    df_gen_info.loc[df_gen_info['EIA Plant Name'] == 'Interstate', '923 Cooling Type'] = \
        'RI'  # Based on regional data availability
    df_gen_info.loc[df_gen_info['EIA Plant Name'] == 'Gibson City Energy Center LLC', '923 Cooling Type'] = \
        'RI'  # Based on regional data availability
    df_gen_info.loc[df_gen_info['EIA Plant Name'] == 'Rantoul', '923 Cooling Type'] = 'OC'
    df_gen_info.loc[df_gen_info['EIA Plant Name'] == 'Tuscola Station', '923 Cooling Type'] = 'OC'
    df_gen_info.loc[df_gen_info['EIA Plant Name'] == 'E D Edwards', '923 Cooling Type'] = 'OC'

    return df_gen_info


def get_regional(df):

    # Convert Units
    df['Withdrawal Rate (Gallon/kWh)'] = \
        df['Water Withdrawal Volume (Million Gallons)'].astype('float64') \
        / df['Gross Generation from Steam Turbines (MWh)'].astype('float64') * 1000  # Convert to Gallon/kWh
    df['Consumption Rate (Gallon/kWh)'] = \
        df['Water Consumption Volume (Million Gallons)'].astype('float64') /\
        df['Gross Generation from Steam Turbines (MWh)'].astype('float64') * 1000  # Convert to Gallon/kWh

    # Substitute Simple Fuel Types
    df['Fuel Type'] = df['Generator Primary Technology'].replace(
        {'Nuclear': 'nuclear',
         'Natural Gas Steam Turbine': 'ng',
         'Conventional Steam Coal': 'coal',
         'Natural Gas Fired Combined Cycle': 'ng',
         'Petroleum Liquids': np.nan})
    df = df[df['Fuel Type'].notna()]

    # Filter to only Illinois Plants
    df = df[df['State'].isin(['IL'])]

    # Filter to only cooling systems in synthetic region (Hardcoded)
    df = df[((df['Fuel Type'] == 'coal') & (df['923 Cooling Type'] == 'RI')) |
            ((df['Fuel Type'] == 'coal') & (df['923 Cooling Type'] == 'RC')) |
            ((df['Fuel Type'] == 'coal') & (df['923 Cooling Type'] == 'OC')) |
            ((df['Fuel Type'] == 'nuclear') & (df['923 Cooling Type'] == 'RC')) |
            ((df['Fuel Type'] == 'ng') & (df['923 Cooling Type'] == 'RI'))]

    # Filter based on real values
    df = df[df['Withdrawal Rate (Gallon/kWh)'].notna()]
    df = df[np.isfinite(df['Withdrawal Rate (Gallon/kWh)'])]
    df = df[df['Consumption Rate (Gallon/kWh)'].notna()]
    df = df[np.isfinite(df['Consumption Rate (Gallon/kWh)'])]

    # Filter based on values that aren't zero
    df = df[df['Withdrawal Rate (Gallon/kWh)'] != 0.0]
    df = df[df['Consumption Rate (Gallon/kWh)'] != 0.0]

    # Filter generators that reported less than 50% of the observations
    df.set_index(['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID'], inplace=True)
    df['Observations'] = 1
    df_sum = df.groupby(['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID']).sum()
    df = df.loc[df_sum[df_sum['Observations'] > 36].index]
    df = df.reset_index()

    # Iglewicz B and Hoaglin D (1993) Page 11 Modified Z-Score Filtering
    df_median = df.groupby(['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID']).median()
    df = df.reset_index()
    df[['Withdrawal Rate (Gallon/kWh) Median', 'Consumption Rate (Gallon/kWh) Median']] = \
        df.join(
            df_median,
            on=['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID'],
            rsuffix=' Median'
        )[['Withdrawal Rate (Gallon/kWh) Median', 'Consumption Rate (Gallon/kWh) Median']]
    df['Withdrawal Rate (Gallon/kWh) Absolute Difference'] = \
        (df['Withdrawal Rate (Gallon/kWh)'] - df['Withdrawal Rate (Gallon/kWh) Median']).abs()
    df['Consumption Rate (Gallon/kWh) Absolute Difference'] = \
        (df['Consumption Rate (Gallon/kWh)'] - df['Consumption Rate (Gallon/kWh) Median']).abs()
    df_mad = df.groupby(['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID']).median()
    df = df.reset_index()
    df[['Withdrawal Rate (Gallon/kWh) MAD', 'Consumption Rate (Gallon/kWh) MAD']] = df.join(
        df_mad,
        on=['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID'],
        rsuffix=' MAD'
    )[['Withdrawal Rate (Gallon/kWh) Absolute Difference MAD', 'Consumption Rate (Gallon/kWh) Absolute Difference MAD']]
    df['Withdrawal Rate (Gallon/kWh) Modified Z Score'] = \
        (0.6745 * (df['Withdrawal Rate (Gallon/kWh)'] - df['Withdrawal Rate (Gallon/kWh) Median'])
         / df['Withdrawal Rate (Gallon/kWh) MAD']).abs()
    df['Consumption Rate (Gallon/kWh) Modified Z Score'] = \
        (0.6745 * (df['Consumption Rate (Gallon/kWh)'] - df['Consumption Rate (Gallon/kWh) Median'])
         / df['Consumption Rate (Gallon/kWh) MAD']).abs()
    df = df[(df['Consumption Rate (Gallon/kWh) Modified Z Score'] < 3.5) &
            (df['Withdrawal Rate (Gallon/kWh) Modified Z Score'] < 3.5)]

    return df


def kmeans_wrapper(df_region, cool_type, n_cluster):
    idxs = df_region.index[(df_region['Fuel Type'] == 'coal') & (df_region['923 Cooling Type'] == cool_type)]
    kmeans = KMeans(
        n_clusters=n_cluster, random_state=1008
    ).fit(df_region.loc[idxs, ['Summer Capacity of Steam Turbines (MW)']].values)
    df_region.loc[idxs, 'Cluster'] = kmeans.labels_
    return kmeans, df_region


def get_cluster(df_geninfo, kmeans, fuel_type, cool_type):
    idxs_geninfo = df_geninfo.index[(df_geninfo['MATPOWER Fuel'] == fuel_type) &
                                    (df_geninfo['923 Cooling Type'] == cool_type)]
    df_geninfo.loc[idxs_geninfo, 'Cluster'] = kmeans.predict(
        df_geninfo.loc[idxs_geninfo, 'MATPOWER Capacity (MW)'].values.reshape(-1, 1)
    )
    return df_geninfo


def get_cluster_data(df_geninfo, df_region, fuel_type, cool_type, cluster):
    # Get regional cluster data
    idxs_region = df_region.index[(df_region['Fuel Type'] == fuel_type) &
                                  (df_region['923 Cooling Type'] == cool_type) & (df_region['Cluster'] == cluster)]
    arr_with = df_region.loc[idxs_region, 'Withdrawal Rate (Gallon/kWh)'].values
    arr_con = df_region.loc[idxs_region, 'Consumption Rate (Gallon/kWh)'].values

    # Add to generator info
    idxs_geninfo = df_geninfo.index[(df_geninfo['MATPOWER Fuel'] == fuel_type) &
                                    (df_geninfo['923 Cooling Type'] == cool_type) & (df_geninfo['Cluster'] == cluster)]
    df_geninfo.loc[idxs_geninfo, 'Withdrawal Rate Cluster Data (Gallon/kWh)'] = df_geninfo.loc[idxs_geninfo].apply(
        lambda row: arr_with, axis=1
    )
    df_geninfo.loc[idxs_geninfo, 'Consumption Rate Cluster Data (Gallon/kWh)'] = df_geninfo.loc[idxs_geninfo].apply(
        lambda row: arr_con, axis=1
    )
    return df_geninfo


def get_generator_water_data(df_region, df_geninfo):
    # K means clustering for coal plants
    kmeans_coal_ri, df_region = kmeans_wrapper(df_region, 'RI', n_cluster=3)
    kmeans_coal_rc, df_region = kmeans_wrapper(df_region, 'RC', n_cluster=2)
    kmeans_coal_oc, df_region = kmeans_wrapper(df_region, 'OC', n_cluster=4)
    df_region['Cluster'] = df_region['Cluster'].fillna(1.0)  # All other fuel types have the same cluster

    # Get cluster of case generators
    df_geninfo = get_cluster(df_geninfo, kmeans_coal_ri, 'coal', 'RI')
    df_geninfo = get_cluster(df_geninfo, kmeans_coal_rc, 'coal', 'RC')
    df_geninfo = get_cluster(df_geninfo, kmeans_coal_oc, 'coal', 'OC')
    df_geninfo['Cluster'] = df_geninfo['Cluster'].fillna(1.0)  # All other fuel types have the same cluster

    # Add cluster data for each case generator
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'RI', 0.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'RI', 1.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'RI', 2.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'RC', 0.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'RC', 1.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'OC', 0.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'OC', 1.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'OC', 2.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'coal', 'OC', 3.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'nuclear', 'RC', 1.0)
    df_geninfo = get_cluster_data(df_geninfo, df_region, 'ng', 'RI', 1.0)
    df_geninfo[['Withdrawal Rate Cluster Data (Gallon/kWh)',
                'Consumption Rate Cluster Data (Gallon/kWh)']] = \
        df_geninfo[['Withdrawal Rate Cluster Data (Gallon/kWh)',
                    'Consumption Rate Cluster Data (Gallon/kWh)']].fillna(0.0)  # No cooling systems get value of 0

    return df_geninfo, df_region


def get_historic_nonuniform_water_coefficients(df_region):
    df_region['Uniform Water Coefficient Consumption'] = \
        df_region['Withdrawal Rate (Gallon/kWh)'] / df_region['Withdrawal Rate (Gallon/kWh) Median']
    df_region['Uniform Water Coefficient Withdrawal'] = \
        df_region['Consumption Rate (Gallon/kWh)'] / df_region['Consumption Rate (Gallon/kWh) Median']
    df = df_region.melt(
        value_vars=['Uniform Water Coefficient Consumption', 'Uniform Water Coefficient Withdrawal'],
        id_vars=['923 Cooling Type', 'Fuel Type'],
        var_name='Exogenous Parameter'
    )
    df['Fuel/Cooling Type'] = df['Fuel Type'] + '/' + df['923 Cooling Type']
    df = df.drop('Exogenous Parameter', axis=1)
    df = df.sort_values('Fuel Type')
    return df


def cooling_system_information(net, df_eia):

    # Convert generator information dataframe (this makes the processing easier)
    df_gen_info = network_to_gen_info(net)

    # Get cooling system from matched EIA generators
    df_gen_info = get_cooling_system(df_eia, df_gen_info)

    # Get regional estimates of water use
    df_region = get_regional(df_eia)

    # Get generator water use
    df_gen_info, df_region = get_generator_water_data(df_region, df_gen_info)

    # Median case
    df_gen_info['Median Withdrawal Rate (Gallon/kWh)'] = df_gen_info.apply(
        lambda row: np.median(row['Withdrawal Rate Cluster Data (Gallon/kWh)']), axis=1
    )
    df_gen_info['Median Consumption Rate (Gallon/kWh)'] = df_gen_info.apply(
        lambda row: np.median(row['Consumption Rate Cluster Data (Gallon/kWh)']), axis=1
    )

    # Add generator information
    net = add_gen_info_to_network(df_gen_info, net)

    # Get historic nonuniform water coefficients
    df_hnwc = get_historic_nonuniform_water_coefficients(df_region)

    return net, df_hnwc, df_region, df_gen_info


def optimization_information(net):
    # Initialize local vars
    gen_types = ['gen', 'sgen', 'ext_grid']
    objective_labs = ['Total Cost ($)',
                      'Generator Cost ($)',
                      'Water Withdrawal (Gallon)',
                      'Water Consumption (Gallon)']
    internal_cost_term_labs = ['Cost Term ($)',
                               'Cost Term ($/MW)',
                               'Cost Term ($/MW^2)']
    uniform_input_factor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)',
                                 'Uniform Loading Coefficient', 'Uniform Water Coefficient']
    internal_water_term_labs = ['Withdrawal Term ($/MW)', 'Consumption Term ($/MW)']
    internal_poly_cost_labs = ['cp0_eur', 'cp1_eur_per_mw', 'cp2_eur_per_mw2']
    withdrawal_rate_labs = []
    consumption_rate_labs = []
    power_labs = []

    for gen_type in gen_types:
        # Storing Pandapower information
        getattr(net, gen_type)['PANDAPOWER Index'] = getattr(net, gen_type).index.tolist()
        getattr(net, gen_type)['PANDAPOWER Type'] = gen_type

        # Dispatching all generators
        getattr(net, gen_type)['in_service'] = True

        generator_indices = 'MATPOWER Generator ' + getattr(net, gen_type)['MATPOWER Index'].astype(str)
        # Exogenous labels
        withdrawal_rate_labs.extend((generator_indices + ' Withdrawal Rate (Gallon/kWh)').tolist())
        consumption_rate_labs.extend((generator_indices + ' Consumption Rate (Gallon/kWh)').tolist())

        # Power labels
        power_labs.extend((generator_indices + ' Power Output (MW)').tolist())

    # Combining exogenous parameter labels
    load_labs = ('PANDAPOWER Bus ' + net.load['bus'].astype(str) + ' Load (MW)').tolist()
    exogenous_labs = ['Withdrawal Weight ($/Gallon)',
                      'Consumption Weight ($/Gallon)'] + withdrawal_rate_labs + consumption_rate_labs + load_labs

    # Line labels
    line_labs = (
            'Line ' + net.line['from_bus'].astype(str) + '-' + net.line['to_bus'].astype(str) + ' Loading (Percent)'
    ).tolist()

    # Output labels
    results_labs = objective_labs + power_labs + line_labs

    # Set attributes
    setattr(net, 'exogenous_labs', exogenous_labs)
    setattr(net, 'objective_labs', objective_labs)
    setattr(net, 'results_labs', results_labs)
    setattr(net, 'uniform_input_factor_labs', uniform_input_factor_labs)

    # Storing optimization terms (these are written during optimization)
    net.poly_cost[internal_water_term_labs] = np.nan
    net.poly_cost[internal_cost_term_labs] = net.poly_cost[internal_poly_cost_labs]
    net.poly_cost[internal_poly_cost_labs] = np.nan

    # Add MATPOWER index to poly_cost
    df_gen_info = network_to_gen_info(net)  # Turn into dataframe to make easier merge
    net.poly_cost['MATPOWER Index'] = net.poly_cost.merge(
        df_gen_info, left_on=['element', 'et'], right_on=['PANDAPOWER Index', 'PANDAPOWER Type']
    )['MATPOWER Index']

    return net


def uniform_input_factor_multiply(c_water, c_load, beta_with, beta_con, beta_load, labs):
    vals = np.concatenate((c_water * beta_with, c_water * beta_con, c_load * beta_load))
    return pd.Series(vals, index=labs)


def get_uniform_search(df_gridspecs):
    # Set Search Values
    w_with_vals = np.linspace(df_gridspecs['Min']['Withdrawal Weight ($/Gallon)'],
                              df_gridspecs['Max']['Withdrawal Weight ($/Gallon)'],
                              df_gridspecs['Number of Steps']['Withdrawal Weight ($/Gallon)'])
    w_con_vals = np.linspace(df_gridspecs['Min']['Consumption Weight ($/Gallon)'],
                             df_gridspecs['Max']['Consumption Weight ($/Gallon)'],
                             df_gridspecs['Number of Steps']['Withdrawal Weight ($/Gallon)'])
    c_load_vals = np.linspace(df_gridspecs['Min']['Uniform Loading Coefficient'],
                              df_gridspecs['Max']['Uniform Loading Coefficient'],
                              df_gridspecs['Number of Steps']['Uniform Loading Coefficient'])
    c_water_vals = np.linspace(df_gridspecs['Min']['Uniform Water Coefficient'],
                               df_gridspecs['Max']['Uniform Water Coefficient'],
                               df_gridspecs['Number of Steps']['Uniform Water Coefficient'])

    # Create Grid
    df_search = pd.DataFrame(
        list(itertools.product(w_with_vals, w_con_vals, c_load_vals, c_water_vals)), columns=df_gridspecs.index
    )

    return df_search


def water_opf(ser_exogenous, net, t):
    # Create DataFrame of loads
    df_load = ser_exogenous[ser_exogenous.index.str.contains('Load')].to_frame('Load (MW)')
    df_load['bus'] = df_load.index.str.extract(r'(\d+)').astype(int).values
    df_load.reset_index(inplace=True, drop=True)

    # Assign Loads
    net.load = net.load.merge(df_load)
    net.load['p_mw'] = net.load['Load (MW)']

    # Create DataFrame of withdrawal and consumption values
    df_withdrawal = \
        ser_exogenous[ser_exogenous.index.str.contains('Withdrawal Rate')].to_frame('Withdrawal Rate (Gallon/kWh)')
    df_withdrawal['MATPOWER Index'] = df_withdrawal.index.str.extract(r'(\d+)').astype(int).values
    df_withdrawal.reset_index(inplace=True, drop=True)

    # Assign withdrawal terms
    net.poly_cost = net.poly_cost.merge(df_withdrawal)

    # Create DataFrame of consumption values
    df_consumption = \
        ser_exogenous[ser_exogenous.index.str.contains('Consumption Rate')].to_frame('Consumption Rate (Gallon/kWh)')
    df_consumption['MATPOWER Index'] = df_consumption.index.str.extract(r'(\d+)').astype(int).values
    df_consumption.reset_index(inplace=True, drop=True)

    # Assign consumption terms
    net.poly_cost = net.poly_cost.merge(df_consumption)

    # Convert water use units (t: minutes * hr/minutes * kw/MW)
    net.poly_cost['Withdrawal Power Rate (Gallon/MW)'] = net.poly_cost['Withdrawal Rate (Gallon/kWh)'] * t
    net.poly_cost['Consumption Power Rate (Gallon/MW)'] = net.poly_cost['Consumption Rate (Gallon/kWh)'] * t

    # Get water terms
    net.poly_cost['Withdrawal Term ($/MW)'] = \
        net.poly_cost['Withdrawal Power Rate (Gallon/MW)'] * ser_exogenous['Withdrawal Weight ($/Gallon)']
    net.poly_cost['Consumption Term ($/MW)'] = \
        net.poly_cost['Consumption Power Rate (Gallon/MW)'] * ser_exogenous['Consumption Weight ($/Gallon)']

    # Assign terms
    net.poly_cost['cp0_eur'] = net.poly_cost['Cost Term ($)']
    net.poly_cost['cp1_eur_per_mw'] =\
        net.poly_cost[['Cost Term ($/MW)', 'Withdrawal Term ($/MW)', 'Consumption Term ($/MW)']].sum(1)
    net.poly_cost['cp2_eur_per_mw2'] = net.poly_cost['Cost Term ($/MW^2)']

    # Run DC OPF
    try:
        pp.rundcopp(net)
    except pp.optimal_powerflow.OPFNotConverged:
        print('OPFNotConverged: DC OPF did not converge returning nan values')

    return net


def get_internal(net):
    # Initialize local vars
    gen_types = ['res_gen', 'res_sgen', 'res_ext_grid']
    df_internal = pd.DataFrame()

    for gen_type in gen_types:
        # Storing pandapower information
        getattr(net, gen_type)['PANDAPOWER Index'] = getattr(net, gen_type).index.tolist()
        getattr(net, gen_type)['PANDAPOWER Type'] = gen_type.split('_', 1)[1]  # Elliminates 'res_'

        # Appending
        df_internal = df_internal.append(getattr(net, gen_type))

    df_internal = df_internal.reset_index()
    return df_internal


def water_opf_wrapper(ser_exogenous, t, net, output_type='numeric'):
    # Initialize local vars
    net = copy.deepcopy(net)  # Copy network so not changed later

    # Run OPF
    net = water_opf(ser_exogenous, net, t)

    if output_type == 'net':
        return net

    elif output_type == 'numeric':
        # Extract internal decisions
        df_internal = get_internal(net)
        df_internal = df_internal.merge(
            net.poly_cost, right_on=['element', 'et'], left_on=['PANDAPOWER Index', 'PANDAPOWER Type']
        )
        generator_output = df_internal['p_mw'].to_list()
        line_loadings = net.res_line['loading_percent'].to_list()

        # Compute objectives
        gen_obj = (df_internal['Cost Term ($)'] +
                   df_internal['p_mw'] * df_internal['Cost Term ($/MW)'] +
                   df_internal['p_mw'] ** 2 * df_internal['Cost Term ($/MW^2)']).sum(min_count=1)
        with_obj = (df_internal['p_mw'] * df_internal['Withdrawal Power Rate (Gallon/MW)']).sum(min_count=1)
        con_obj = (df_internal['p_mw'] * df_internal['Consumption Power Rate (Gallon/MW)']).sum(min_count=1)
        cos_obj = gen_obj + \
            ser_exogenous['Withdrawal Weight ($/Gallon)'] * with_obj + \
            ser_exogenous['Consumption Weight ($/Gallon)'] * con_obj

        # Formatting Export
        vals = [cos_obj, gen_obj, with_obj, con_obj] + generator_output + line_loadings
        ser_results = pd.Series(vals, index=net.results_labs)

        return ser_results


def uniform_sa(net, n_tasks, n_steps):
    # Initialize local vars
    df_gridspecs = pd.DataFrame(
        data=[[0.0, 0.1, n_steps], [0.0, 1.0, n_steps], [1.0, 1.5, n_steps], [0.5, 1.5, n_steps]],
        index=net.uniform_input_factor_labs,
        columns=['Min', 'Max', 'Number of Steps']
    )
    t = 5 * 1 / 60 * 1000  # minutes * hr/minutes * kw/MW
    print('Success: Initialized Uniform Sensitivity Analysis')

    # Get input factor search space
    df_search = get_uniform_search(df_gridspecs)
    print('Number of Searches: ', len(df_search))
    print('Success: Grid Created')

    # Convert generator information dataframe (this makes the processing easier)
    df_gen_info = network_to_gen_info(net)

    # Multiply median exogenous coefficient values by input factors
    beta_with_median = df_gen_info['Median Withdrawal Rate (Gallon/kWh)'].values
    beta_con_median = df_gen_info['Median Consumption Rate (Gallon/kWh)'].values
    beta_load_median = net.load['p_mw'].values
    print('Success: Uniform SA, Extract Medians')
    df_exogenous = df_search.apply(lambda row: uniform_input_factor_multiply(
        row['Uniform Water Coefficient'],
        row['Uniform Loading Coefficient'],
        beta_with_median,
        beta_con_median,
        beta_load_median,
        net.exogenous_labs[2:]  # First two parameters in df_search
    ), axis=1)
    print('Success: Uniform SA, Multiply exogenous')

    # Combine input factor grid and exogenous parameters
    df_search_exogenous = pd.concat([df_search, df_exogenous], axis=1)

    # Run Sampling
    ddf_search_exogenous = dd.from_pandas(df_search_exogenous, npartitions=n_tasks)
    print('Success: Uniform SA, Dask Conversion')
    df_results = ddf_search_exogenous.apply(
        lambda row: water_opf_wrapper(row[net.exogenous_labs], t, net),
        axis=1,
        meta=pd.DataFrame(columns=net.results_labs, dtype='float64')
    ).compute(scheduler='processes')
    print('Success: Uniform SA, Grid Sampled')
    df_uniform = pd.concat([df_search_exogenous, df_results], axis=1)
    df_uniform = df_uniform.drop_duplicates()  # Sometimes the parallel jobs replicate rows

    return df_uniform


def get_plant_output_ratio(df, df_gen_info, uniform_factor_labs, obj_labs):
    # Internal vars
    n_runs = len(df)
    plant_names = df_gen_info['Plant Name'].unique().tolist()

    # Get generator output
    df_power_output = df.loc[:, df.columns.str.contains('Power Output')]
    df_power_output.columns = df_power_output.columns.str.extract(r'(\d+)').astype(int)[0].tolist()
    df_power_output = df_power_output.transpose()

    # Combine into plants
    df_power_output = df_power_output.merge(df_gen_info, left_index=True, right_on='MATPOWER Index')
    df_plant_capacity_ratio = df_power_output.groupby(['Plant Name']).sum()

    # Get Ratio
    df_plant_capacity_ratio = df_plant_capacity_ratio.iloc[:, 0:n_runs].divide(
        df_plant_capacity_ratio['MATPOWER Capacity (MW)'],
        axis='index'
    )  # Hardcoded

    # Combine with input factors
    df_plant_capacity_ratio = df_plant_capacity_ratio.transpose()
    df_plant_capacity_ratio = df_plant_capacity_ratio.join(df[uniform_factor_labs+obj_labs])

    # Get Cooling System Type
    df_plant_capacity_ratio = pd.melt(
        df_plant_capacity_ratio,
        value_vars=plant_names,
        id_vars=uniform_factor_labs + obj_labs,
        var_name='Plant Name',
        value_name='Output'
    )
    df_plant_capacity_ratio = df_plant_capacity_ratio.merge(
        df_gen_info[['Plant Name', '923 Cooling Type', 'MATPOWER Fuel']])

    return df_plant_capacity_ratio


def get_line_flow_difference(net):
    # Preparing information (could go in optimization_information later)
    net = copy.deepcopy(net)
    net.line['name'] = net.line['from_bus'].astype(str) + ' - ' + net.line['to_bus'].astype(str)
    net.trafo['name'] = net.trafo['hv_bus'].astype(str) + ' - ' + net.trafo['lv_bus'].astype(str)

    # Get uniform df for subsets (See uniform_sa for references)
    t = 5 * 1 / 60 * 1000  # minutes * hr/minutes * kw/MW
    df_search = pd.DataFrame({'Uniform Water Coefficient': [0.5, 0.5],
                              'Uniform Loading Coefficient': [1.5, 1.5],
                              'Withdrawal Weight ($/Gallon)': [0.0, 0.1],
                              'Consumption Weight ($/Gallon)': [0.0, 0.0]})
    df_gen_info = network_to_gen_info(net)
    beta_with_median = df_gen_info['Median Withdrawal Rate (Gallon/kWh)'].values
    beta_con_median = df_gen_info['Median Consumption Rate (Gallon/kWh)'].values
    beta_load_median = net.load['p_mw'].values
    df_exogenous = df_search.apply(lambda row: uniform_input_factor_multiply(
        row['Uniform Water Coefficient'],
        row['Uniform Loading Coefficient'],
        beta_with_median,
        beta_con_median,
        beta_load_median,
        net.exogenous_labs[2:]  # First two parameters in df_search
    ), axis=1)
    df_search_exogenous = pd.concat([df_search, df_exogenous], axis=1)
    list_net = list(df_search_exogenous.apply(
        lambda row: water_opf_wrapper(row[net.exogenous_labs], t, net, output_type='net'),
        axis=1
    ))

    # Label flows
    net_diff = copy.deepcopy(list_net[0])
    net_diff.res_line['Traditional OPF Loading (Percent)'] = list_net[0].res_line['loading_percent']
    net_diff.res_line['Water OPF Loading (Percent)'] = list_net[1].res_line['loading_percent']
    net_diff.res_trafo['Traditional OPF Loading (Percent)'] = list_net[0].res_trafo['loading_percent']
    net_diff.res_trafo['Water OPF Loading (Percent)'] = list_net[1].res_trafo['loading_percent']

    # Difference in flow
    net_diff.res_line['Change in Loading (Percent)'] = \
        net_diff.res_line['Water OPF Loading (Percent)'] - net_diff.res_line['Traditional OPF Loading (Percent)']
    net_diff.res_trafo['Change in Loading (Percent)'] = \
        net_diff.res_trafo['Water OPF Loading (Percent)'] - net_diff.res_trafo['Traditional OPF Loading (Percent)']
    # Tabulate line flows
    net_diff.res_line['name'] = net_diff.line['name']
    net_diff.res_trafo['name'] = net_diff.trafo['name']
    cols = ['name', 'Traditional OPF Loading (Percent)', 'Water OPF Loading (Percent)', 'Change in Loading (Percent)']
    df_line_flows = pd.concat([net_diff.res_line[cols], net_diff.res_trafo[cols]])
    df_line_flows = df_line_flows.sort_values('Change in Loading (Percent)')

    return net_diff, df_line_flows


def fit_single_models(df, obj_labs, factor_labs):
    mods = {}
    for i in obj_labs:
        clf = tree.DecisionTreeRegressor(random_state=1008, max_depth=5, max_leaf_nodes=12)
        mods[i] = clf.fit(X=df[factor_labs], y=df[i])
    return mods
