import os
import copy
import itertools

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import dask.dataframe as dd
import pandapower as pp
from dtreeviz.trees import *  # Requires pip version
from svglib.svglib import svg2rlg



def grid_setup(net):
    # Dispatching all generators
    net.sgen['in_service'] = True
    net.gen['in_service'] = True
    net.ext_grid['in_service'] = True

    return net


def generator_match(df_gen_info, df_gen_matches):

    # Merge manual matches
    df_gen_info_match = df_gen_info.merge(df_gen_matches)

    # Anonymous Plant Names
    powerworld_plants = df_gen_info_match['POWERWORLD Plant Name'].unique()
    anonymous_plants = [f'Plant {i}' for i in range(1, len(powerworld_plants) + 1)]
    d = dict(zip(powerworld_plants, anonymous_plants))
    df_gen_info_match['Plant Name'] = df_gen_info_match['POWERWORLD Plant Name'].map(d)

    return df_gen_info_match


def import_EIA(pathto_EIA):
    # Local Vars
    years = ['2019', '2018', '2017', '2016', '2015', '2014']
    df_list = []
    # Import all dataframes
    for i in years:
        path = os.path.join(pathto_EIA, 'cooling_detail_' + i + '.xlsx')
        print(i)
        # Import Dataframe
        df_temp = pd.read_excel(path, header=2)
        # Replace space values with nan values
        df_temp = df_temp.replace(r'^\s*$', np.nan, regex=True)
        df_list.append(df_temp)
    # Concat Dataframes into Single Dataframe
    df = pd.concat(df_list)
    return df


def getCoolingSystem(df_EIA, df_geninfo):
    # Matches from manual analysis of EIA dataset
    df_EIA = df_EIA.drop_duplicates(subset='Plant Name', keep='first')
    df_geninfo['923 Cooling Type'] = df_geninfo.merge(df_EIA, right_on='Plant Name', left_on='EIA Plant Name', how='left')['923 Cooling Type']
    # Assumptions about cooling systems
    df_geninfo.loc[df_geninfo['MATPOWER Fuel'] == 'wind', '923 Cooling Type'] = 'No Cooling System'  # Wind is assumed to not use water
    df_geninfo.loc[(df_geninfo['MATPOWER Type'] == 'GT') & (df_geninfo['MATPOWER Fuel'] == 'ng') & (df_geninfo['MATPOWER Capacity (MW)'] < 30), '923 Cooling Type'] = 'No Cooling System'  # Assume Small Capacity Natural Gas Turbines Don't Have Cooling System
    # One off matching based on searching
    df_geninfo.loc[df_geninfo['EIA Plant Name'] == 'Interstate', '923 Cooling Type'] = 'RI'  # Based on regional data availability
    df_geninfo.loc[df_geninfo['EIA Plant Name'] == 'Gibson City Energy Center LLC', '923 Cooling Type'] = 'RI'  # Based on regional data availability
    df_geninfo.loc[df_geninfo['EIA Plant Name'] == 'Rantoul', '923 Cooling Type'] = 'OC'
    df_geninfo.loc[df_geninfo['EIA Plant Name'] == 'Tuscola Station', '923 Cooling Type'] = 'OC'
    df_geninfo.loc[df_geninfo['EIA Plant Name'] == 'E D Edwards', '923 Cooling Type'] = 'OC'
    return df_geninfo


def getRegional(df):
    # Convert Units
    df['Withdrawal Rate (Gallon/kWh)'] = df['Water Withdrawal Volume (Million Gallons)'].astype('float64') / df['Gross Generation from Steam Turbines (MWh)'].astype('float64') * 1000  # Convert to Gallon/kWh
    df['Consumption Rate (Gallon/kWh)'] = df['Water Consumption Volume (Million Gallons)'].astype('float64') / df['Gross Generation from Steam Turbines (MWh)'].astype('float64') * 1000  # Convert to Gallon/kWh
    # Substitute Simple Fuel Types
    df['Fuel Type'] = df['Generator Primary Technology'].replace({'Nuclear': 'nuclear',
                                                                  'Natural Gas Steam Turbine': 'ng',
                                                                  'Conventional Steam Coal': 'coal',
                                                                  'Natural Gas Fired Combined Cycle': 'ng',
                                                                  'Petroleum Liquids': np.nan})
    df = df[df['Fuel Type'].notna()]
    # Filter to only Illinois Plants
    df = df[df['State'].isin(['IL'])]
    # Filter to only cooling systems in synthetic region (Hardcoded)
    df = df[((df['Fuel Type'] == 'coal') & (df['923 Cooling Type'] == 'RI')) |\
            ((df['Fuel Type'] == 'coal') & (df['923 Cooling Type'] == 'RC')) |\
            ((df['Fuel Type'] == 'coal') & (df['923 Cooling Type'] == 'OC')) |\
            ((df['Fuel Type'] == 'nuclear') & (df['923 Cooling Type'] == 'RC')) |\
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
    df[['Withdrawal Rate (Gallon/kWh) Median', 'Consumption Rate (Gallon/kWh) Median']] = df.join(df_median, on=['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID'], rsuffix=' Median')[['Withdrawal Rate (Gallon/kWh) Median', 'Consumption Rate (Gallon/kWh) Median']]
    df['Withdrawal Rate (Gallon/kWh) Absolute Difference'] = (df['Withdrawal Rate (Gallon/kWh)'] - df['Withdrawal Rate (Gallon/kWh) Median']).abs()
    df['Consumption Rate (Gallon/kWh) Absolute Difference'] = (df['Consumption Rate (Gallon/kWh)'] - df['Consumption Rate (Gallon/kWh) Median']).abs()
    df_MAD = df.groupby(['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID']).median()
    df = df.reset_index()
    df[['Withdrawal Rate (Gallon/kWh) MAD', 'Consumption Rate (Gallon/kWh) MAD']] = df.join(df_MAD, on=['Plant Name', 'Generator ID', 'Boiler ID', 'Cooling ID'], rsuffix=' MAD')[['Withdrawal Rate (Gallon/kWh) Absolute Difference MAD', 'Consumption Rate (Gallon/kWh) Absolute Difference MAD']]
    df['Withdrawal Rate (Gallon/kWh) Modified Z Score'] = (0.6745 * (df['Withdrawal Rate (Gallon/kWh)'] - df['Withdrawal Rate (Gallon/kWh) Median'])/df['Withdrawal Rate (Gallon/kWh) MAD']).abs()
    df['Consumption Rate (Gallon/kWh) Modified Z Score'] = (0.6745 * (df['Consumption Rate (Gallon/kWh)'] - df['Consumption Rate (Gallon/kWh) Median']) / df['Consumption Rate (Gallon/kWh) MAD']).abs()
    df = df[(df['Consumption Rate (Gallon/kWh) Modified Z Score'] < 3.5) & (df['Withdrawal Rate (Gallon/kWh) Modified Z Score'] < 3.5)]
    return df


def kmeansWapper(df_region, cool_type, n_cluster):
    idxs = df_region.index[(df_region['Fuel Type'] == 'coal') & (df_region['923 Cooling Type'] == cool_type)]
    kmeans = KMeans(n_clusters=n_cluster, random_state=1008).fit(df_region.loc[idxs, ['Summer Capacity of Steam Turbines (MW)']].values)
    df_region.loc[idxs, 'Cluster'] = kmeans.labels_
    return kmeans, df_region


def getCluster(df_geninfo, kmeans, fuel_type, cool_type):
    idxs_geninfo = df_geninfo.index[(df_geninfo['MATPOWER Fuel'] == fuel_type) & (df_geninfo['923 Cooling Type'] == cool_type)]
    df_geninfo.loc[idxs_geninfo, 'Cluster'] = kmeans.predict(df_geninfo.loc[idxs_geninfo, 'MATPOWER Capacity (MW)'].values.reshape(-1, 1))
    return df_geninfo


def getClusterData(df_geninfo, df_region, fuel_type, cool_type, cluster):
    # Get regional cluster data
    idxs_region = df_region.index[(df_region['Fuel Type'] == fuel_type) & (df_region['923 Cooling Type'] == cool_type) & (df_region['Cluster'] == cluster)]
    arr_with = df_region.loc[idxs_region, 'Withdrawal Rate (Gallon/kWh)'].values
    arr_con = df_region.loc[idxs_region, 'Consumption Rate (Gallon/kWh)'].values
    # Add to generator info
    idxs_geninfo = df_geninfo.index[(df_geninfo['MATPOWER Fuel'] == fuel_type) & (df_geninfo['923 Cooling Type'] == cool_type) & (df_geninfo['Cluster'] == cluster)]
    df_geninfo.loc[idxs_geninfo, 'Withdrawal Rate Cluster Data (Gallon/kWh)'] = df_region.loc[idxs_geninfo].apply(lambda row: arr_with, axis=1)
    df_geninfo.loc[idxs_geninfo, 'Consumption Rate Cluster Data (Gallon/kWh)'] = df_region.loc[idxs_geninfo].apply(lambda row: arr_con, axis=1)
    return df_geninfo


def getGeneratorWaterData(df_region, df_geninfo):
    # K means clustering for coal plants
    kmeans_coal_RI, df_region = kmeansWapper(df_region, 'RI', n_cluster=3)
    kmeans_coal_RC, df_region = kmeansWapper(df_region, 'RC', n_cluster=2)
    kmeans_coal_OC, df_region = kmeansWapper(df_region, 'OC', n_cluster=4)
    df_region['Cluster'] = df_region['Cluster'].fillna(1.0)  # All other fuel types have the same cluster
    # Get cluster of case generators
    df_geninfo = getCluster(df_geninfo, kmeans_coal_RI, 'coal', 'RI')
    df_geninfo = getCluster(df_geninfo, kmeans_coal_RC, 'coal', 'RC')
    df_geninfo = getCluster(df_geninfo, kmeans_coal_OC, 'coal', 'OC')
    df_geninfo['Cluster'] = df_geninfo['Cluster'].fillna(1.0)  # All other fuel types have the same cluster
    # Add cluster data for each case generator
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'RI', 0.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'RI', 1.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'RI', 2.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'RC', 0.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'RC', 1.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'OC', 0.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'OC', 1.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'OC', 2.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'coal', 'OC', 3.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'nuclear', 'RC', 1.0)
    df_geninfo = getClusterData(df_geninfo, df_region, 'ng', 'RI', 1.0)
    df_geninfo[['Withdrawal Rate Cluster Data (Gallon/kWh)', 'Consumption Rate Cluster Data (Gallon/kWh)']] = \
        df_geninfo[['Withdrawal Rate Cluster Data (Gallon/kWh)', 'Consumption Rate Cluster Data (Gallon/kWh)']].fillna(0.0)  # No cooling systems get assigned a value of 0
    return df_geninfo, df_region


def get_historic_nonuniform_water_coefficients(df_region, df_gen_info_match_water):
    df_region['Uniform Water Coefficient Consumption'] = df_region['Withdrawal Rate (Gallon/kWh)'] / df_region['Withdrawal Rate (Gallon/kWh) Median']
    df_region['Uniform Water Coefficient Withdrawal'] = df_region['Consumption Rate (Gallon/kWh)'] / df_region['Consumption Rate (Gallon/kWh) Median']
    df = df_region.melt(value_vars=['Uniform Water Coefficient Consumption', 'Uniform Water Coefficient Withdrawal'],
                      id_vars=['923 Cooling Type', 'Fuel Type'], var_name='Exogenous Parameter')
    df['Fuel/Cooling Type'] = df['Fuel Type'] + '/' + df['923 Cooling Type']
    df = df.drop('Exogenous Parameter', axis=1)
    df = df.sort_values('Fuel Type')
    return df


def regionDistribututionPlotter(df):
    data = (df['Withdrawal Rate (Gallon/kWh)'] / df['Withdrawal Rate (Gallon/kWh) Median']).append(
        df['Consumption Rate (Gallon/kWh)'] / df['Consumption Rate (Gallon/kWh) Median'])
    g = sns.histplot(data)
    plt.show()
    return g


def regionBoxPlotter(df):
    df_plot = df.melt(value_vars=['Withdrawal Rate (Gallon/kWh)', 'Consumption Rate (Gallon/kWh)'],
                      id_vars=['923 Cooling Type', 'Fuel Type'], var_name='Exogenous Parameter')
    g = sns.catplot(data=df_plot, x='value', y='923 Cooling Type', col='Exogenous Parameter', row='Fuel Type',
                     kind='box', sharex='col', height=3, aspect=1.2)
    g.axes[0, 0].set_title('Coal')
    g.axes[0, 1].set_title('Coal')
    g.axes[1, 0].set_title('Nuclear')
    g.axes[1, 1].set_title('Nuclear')
    g.axes[2, 0].set_title('Natural Gas')
    g.axes[2, 1].set_title('Natural Gas')
    g.axes[2, 0].set_xlabel(r'Withdrawal Rate $\beta_{with}$ [Gallon/kWh]')
    g.axes[2, 1].set_xlabel(r'Consumption Rate $\beta_{con}$ [Gallon/kWh]')
    plt.tight_layout()
    plt.show()
    return g


def hnwc_plotter(df, df_gen_info):

    # Titles
    df_titles = df_gen_info.sort_values('MATPOWER Fuel')
    df_titles['Fuel/Cooling Type'] = df_titles['MATPOWER Fuel'] + '/' + df_titles['923 Cooling Type']
    df_titles = df_titles.groupby('Fuel/Cooling Type')['Plant Name'].apply(set).reset_index(name='Plants')
    df_titles['Grouped Plants'] = df_titles.apply(
        lambda row: 'Plant ID: ' + ", ".join([item for subitem in row['Plants'] for item in subitem.split() if item.isdigit()]),
        axis=1
    )
    df_titles['Title'] = df_titles['Fuel/Cooling Type'] + ' (' + df_titles['Grouped Plants'] + ')'

    # Plotting
    g = sns.FacetGrid(df, col='Fuel/Cooling Type', col_wrap=2, sharex=False, sharey=False, aspect=1.5)
    g.map(sns.histplot, 'value', stat='density')
    for ax in g.axes:
        fuel_cool = ax.get_title().split('Fuel/Cooling Type = ', 1)[1]
        ax.set_title(df_titles[df_titles['Fuel/Cooling Type'] == fuel_cool]['Title'].values[0])
    g.set_xlabels('Non-Uniform Water Coefficient')
    plt.show()
    return g


def coalPlotter(df):
    df = df[df['Fuel Type'] == 'coal']
    df_plot = df.melt(value_vars=['Withdrawal Rate (Gallon/kWh)', 'Consumption Rate (Gallon/kWh)'],
                      id_vars=['923 Cooling Type', 'Fuel Type', 'Summer Capacity of Steam Turbines (MW)', 'Cluster'],
                      var_name='Exogenous Parameter')
    g = sns.FacetGrid(df_plot, col='Exogenous Parameter', row='923 Cooling Type', hue='Cluster', sharex='col',
                      height=3, aspect=1.2)
    g.map(sns.scatterplot, 'value', 'Summer Capacity of Steam Turbines (MW)')
    g.axes[0, 0].set_title('RI')
    g.axes[0, 1].set_title('RI')
    g.axes[1, 0].set_title('RC')
    g.axes[1, 1].set_title('RC')
    g.axes[2, 0].set_title('OC')
    g.axes[2, 1].set_title('OC')
    g.set_ylabels('Capacity (MW)')
    g.axes[2, 0].set_xlabel(r'Withdrawal Rate $\beta_{with}$ [Gallon/kWh]')
    g.axes[2, 1].set_xlabel(r'Consumption Rate $\beta_{con}$ [Gallon/kWh]')
    g.add_legend()
    plt.show()
    return g


def cooling_system_information(df_gen_info_match, df_EIA):
    # Get Cooling System From Matched EIA Generators
    df_gen_info_match_water = getCoolingSystem(df_EIA, df_gen_info_match)

    # Get Regional Estimates of Water Use
    df_region = getRegional(df_EIA)

    # Get Generator Water Use
    df_gen_info_match_water, df_region = getGeneratorWaterData(df_region, df_gen_info_match_water)

    # Median Case
    df_gen_info_match_water['Median Withdrawal Rate (Gallon/kWh)'] = df_gen_info_match_water.apply(lambda row: np.median(row['Withdrawal Rate Cluster Data (Gallon/kWh)']), axis=1)
    df_gen_info_match_water['Median Consumption Rate (Gallon/kWh)'] = df_gen_info_match_water.apply(lambda row: np.median(row['Consumption Rate Cluster Data (Gallon/kWh)']), axis=1)

    # Get historic nonuniform water coefficients
    df_hnwc = get_historic_nonuniform_water_coefficients(df_region, df_gen_info_match_water)

    # Plotting
    fig_regionDistribututionPlotter = regionDistribututionPlotter(df_region).figure
    fig_regionBoxPlotter = regionBoxPlotter(df_region).fig
    fig_coalPlotter = coalPlotter(df_region).fig
    fig_hnwc_plotter = hnwc_plotter(df_hnwc, df_gen_info_match_water).fig

    return df_gen_info_match_water, df_hnwc, fig_regionDistribututionPlotter, fig_regionBoxPlotter, fig_coalPlotter, fig_hnwc_plotter

def uniformFactorMultiply(c_water, c_load, beta_with, beta_con, beta_load, exogenous_labs):
    vals = np.concatenate((c_water * beta_with, c_water * beta_con, c_load * beta_load))
    idxs = exogenous_labs[2:]
    return pd.Series(vals, index=idxs)


def getSearch(df_gridspecs, df_geninfo, net, exogenous_labs):
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
    df_search = pd.DataFrame(list(itertools.product(w_with_vals, w_con_vals, c_load_vals, c_water_vals)), columns=df_gridspecs.index)
    # Multiply exogenous parameters
    beta_with = df_geninfo['Median Withdrawal Rate (Gallon/kWh)'].values
    beta_con = df_geninfo['Median Consumption Rate (Gallon/kWh)'].values
    beta_load = net.load['p_mw'].values
    df_exogenous = df_search.apply(lambda row: uniformFactorMultiply(row['Uniform Water Coefficient'], row['Uniform Loading Coefficient'], beta_with, beta_con, beta_load, exogenous_labs), axis=1)
    # Combine
    df_search = pd.concat([df_search, df_exogenous], axis=1)
    return df_search


def matchIndex(df_geninfo, net):
    # Initialize
    df_gen = pd.DataFrame()
    df_sgen = pd.DataFrame()
    df_ext_grid = pd.DataFrame()
    # Generators
    df_gen['MATPOWER Index'] = net.gen['bus'] + 1
    df_gen['PANDAPOWER Index'] = net.gen['bus'].index.to_list()
    df_gen['PANDAPOWER Bus Type'] = 'gen'
    # Static Generators
    df_sgen['MATPOWER Index'] = net.sgen['bus'] + 1
    df_sgen['PANDAPOWER Index'] = net.sgen['bus'].index.to_list()
    df_sgen['PANDAPOWER Bus Type'] = 'sgen'
    # External Grid
    df_ext_grid['MATPOWER Index'] = net.ext_grid['bus'] + 1
    df_ext_grid['PANDAPOWER Index'] = net.ext_grid['bus'].index.to_list()
    df_ext_grid['PANDAPOWER Bus Type'] = 'ext_grid'
    # Combine
    df_match = pd.concat([df_gen, df_sgen, df_ext_grid])
    df_match.reset_index()
    # Join
    df_geninfo = df_geninfo.merge(df_match, on='MATPOWER Index')
    return df_geninfo


def waterOPF(ser_exogenous, t, results_labs, net, df_geninfo):
    # Initialize
    df_geninfo = df_geninfo.copy()
    net = copy.deepcopy(net)  # Copy network so not changed later
    obj_colnames = ['cp0_eur', 'cp1_eur_per_mw', 'cp2_eur_per_mw2']
    idx_colnames = ['element', 'et']
    # Create DataFrame of loads
    df_load = ser_exogenous[ser_exogenous.index.str.contains('Load')].to_frame('Load (MW)')
    df_load['bus'] = df_load.index.str.extract('(\d+)').astype(int).values
    df_load.reset_index(inplace=True, drop=True)
    # Create DataFrame of cost values
    df_cost = net.poly_cost[idx_colnames + obj_colnames]
    # Create DataFrame of withdrawal values
    df_withdrawal = ser_exogenous[ser_exogenous.index.str.contains('Withdrawal Rate')].to_frame('Withdrawal Rate (Gallon/kWh)')
    df_withdrawal['MATPOWER Index'] = df_withdrawal.index.str.extract('(\d+)').astype(int).values
    df_withdrawal.reset_index(inplace=True, drop=True)
    # Create DataFrame of consumption values
    df_consumption = ser_exogenous[ser_exogenous.index.str.contains('Consumption Rate')].to_frame('Consumption Rate (Gallon/kWh)')
    df_consumption['MATPOWER Index'] = df_consumption.index.str.extract('(\d+)').astype(int).values
    df_consumption.reset_index(inplace=True, drop=True)
    # Write objectives information to generator information
    df_geninfo[['Cost Term ($)', 'Cost Term ($/MW)', 'Cost Term ($/MW^2)']] = df_geninfo.merge(df_cost, left_on=['PANDAPOWER Index', 'PANDAPOWER Bus Type'], right_on=idx_colnames)[obj_colnames]
    df_geninfo['Withdrawal Rate (Gallon/kWh)'] = df_geninfo.merge(df_withdrawal, on='MATPOWER Index')['Withdrawal Rate (Gallon/kWh)']
    df_geninfo['Consumption Rate (Gallon/kWh)'] = df_geninfo.merge(df_consumption, on='MATPOWER Index')['Consumption Rate (Gallon/kWh)']
    # Convert Units
    df_geninfo['Withdrawal Rate (Gallon/MW)'] = df_geninfo['Withdrawal Rate (Gallon/kWh)'] * t  # minutes * hr/minutes * kw/MW
    df_geninfo['Consumption Rate (Gallon/MW)'] = df_geninfo['Consumption Rate (Gallon/kWh)'] * t  # minutes * hr/minutes * kw/MW
    # Combine and weight objectives
    df_geninfo['Weighted Linear Term'] = df_geninfo['Cost Term ($/MW)'] + df_geninfo['Withdrawal Rate (Gallon/MW)'] * ser_exogenous['Withdrawal Weight ($/Gallon)'] + df_geninfo['Consumption Rate (Gallon/MW)'] * ser_exogenous['Consumption Weight ($/Gallon)']
    # Assign exogenous parameters to case
    net.poly_cost[obj_colnames] = net.poly_cost.merge(df_geninfo, left_on=idx_colnames, right_on=['PANDAPOWER Index', 'PANDAPOWER Bus Type'])[['Cost Term ($)', 'Weighted Linear Term', 'Cost Term ($/MW^2)']]
    net.load['p_mw'] = net.load.merge(df_load)['Load (MW)']
    # Run DC OPF
    try:
        pp.rundcopp(net)
        state = 'converge'
    except:
        state = 'not converge'
    # Output Depended on State
    if state == 'converge':
        # Extract internal decisions (power output)
        for type in ['gen', 'sgen', 'ext_grid']:
            idxs = df_geninfo.index[df_geninfo['PANDAPOWER Bus Type'] == type]
            df_geninfo.loc[idxs, 'Power Output (MW)'] = df_geninfo.iloc[idxs, :].merge(net['res_'+type], left_on='PANDAPOWER Index', right_index=True)['p_mw']
        # Compute Capacity Ratios
        df_geninfo['Ratio of Capacity'] = df_geninfo['Power Output (MW)'] / df_geninfo['MATPOWER Capacity (MW)']
        # Compute Objectives
        F_gen = (df_geninfo['Cost Term ($)'] + df_geninfo['Power Output (MW)'] * df_geninfo['Cost Term ($/MW)'] + df_geninfo['Power Output (MW)']**2 * df_geninfo['Cost Term ($/MW^2)']).sum()
        F_with = (df_geninfo['Power Output (MW)'] * df_geninfo['Withdrawal Rate (Gallon/MW)']).sum()
        F_con = (df_geninfo['Power Output (MW)'] * df_geninfo['Consumption Rate (Gallon/MW)']).sum()
        F_cos = F_gen + ser_exogenous['Withdrawal Weight ($/Gallon)']*F_with + ser_exogenous['Consumption Weight ($/Gallon)']*F_con
        internal_decs = df_geninfo['Ratio of Capacity'].to_list()
    elif state == 'not converge':
        F_cos = F_with = F_con = F_gen = np.nan
        internal_decs = [np.nan]*len(df_geninfo)
    return pd.Series([F_cos, F_gen, F_with, F_con] + internal_decs, index=results_labs)


def uniform_sa(df_gen_info_match_water, net, n_tasks, n_steps, uniform_factor_labs, obj_labs):
    # Initialize
    exogenous_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)'] + \
                     ('MATPOWER Generator ' + df_gen_info_match_water['MATPOWER Index'].astype(str) + ' Withdrawal Rate (Gallon/kWh)').tolist() + \
                     ('MATPOWER Generator ' + df_gen_info_match_water['MATPOWER Index'].astype(str) + ' Consumption Rate (Gallon/kWh)').tolist() + \
                     ('PANDAPOWER Bus ' + net.load['bus'].astype(str) + ' Load (MW)').tolist()
    results_labs = obj_labs + ('MATPOWER Generator ' + df_gen_info_match_water['MATPOWER Index'].astype(str) + ' Ratio of Capacity').to_list()
    df_gridspecs = pd.DataFrame(data=[[0.0, 0.1, n_steps], [0.0, 1.0, n_steps], [1.0, 1.5, n_steps], [0.5, 1.5, n_steps]],
                                index=uniform_factor_labs, columns=['Min', 'Max', 'Number of Steps'])
    t = 5 * 1 / 60 * 1000  # minutes * hr/minutes * kw/MW
    print('Success: Initialized')
    # Get Pandapower indices
    df_gen_info_match_water = matchIndex(df_gen_info_match_water, net)
    # Get Grid
    df_search = getSearch(df_gridspecs, df_gen_info_match_water, net, exogenous_labs)
    print('Number of Searches: ', len(df_search))
    print('Success: Grid Created')
    # Run Search
    ddf_search = dd.from_pandas(df_search, npartitions=n_tasks)
    df_results = ddf_search.apply(lambda row: waterOPF(row[exogenous_labs], t, results_labs, net, df_gen_info_match_water), axis=1, meta={key:'float64' for key in results_labs}).compute(scheduler='processes')
    df_uniform = pd.concat([df_results, df_search], axis=1)
    df_uniform.drop_duplicates()  # Sometimes the parallel jobs replicate rows
    print('Success: Grid Searched')
    return df_uniform


def viz_effect_of_withdrawal_weight(df, uniform_factor_labs, obj_labs):

    # Subsetting data
    plot_df = df[uniform_factor_labs + obj_labs]
    plot_df = plot_df[plot_df['Consumption Weight ($/Gallon)'] == 0.0]
    plot_df = plot_df[plot_df['Withdrawal Weight ($/Gallon)'].isin([0.0, 0.1])]
    plot_df = plot_df.round({'Uniform Water Coefficient': 2})
    plot_df = plot_df[plot_df['Uniform Water Coefficient'].isin(
        plot_df['Uniform Water Coefficient'].unique()[1::2])]  # Select every other uniform water coefficient

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    g = sns.lineplot(data=plot_df, x='Uniform Loading Coefficient', y='Water Withdrawal (Gallon)',
                     hue='Uniform Water Coefficient', style='Withdrawal Weight ($/Gallon)', markers=True, ax=ax)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return fig


def get_plant_output_ratio(df, df_gen_info, uniform_factor_labs, obj_labs):
    # Internal vars
    n_runs = len(df)
    plant_names = df_gen_info['Plant Name'].unique().tolist()

    # Get generator output
    df_capacity_ratio = df.loc[:, df.columns.str.contains('Ratio of Capacity')]
    df_capacity_ratio.columns = df_capacity_ratio.columns.str.extract('(\d+)').astype(int)[0].tolist()
    df_capacity_ratio = df_capacity_ratio.transpose()
    ser_gen_capacity = pd.Series(df_gen_info['MATPOWER Capacity (MW)'].values, index=df_gen_info['MATPOWER Index'])
    df_gen_output = df_capacity_ratio.multiply(ser_gen_capacity, axis='index')

    # Combine into plants
    df_gen_output = df_gen_output.merge(df_gen_info, left_index=True, right_on='MATPOWER Index')
    df_plant_capacity_ratio = df_gen_output.groupby(['Plant Name']).sum()

    # Get Ratio
    df_plant_capacity_ratio = df_plant_capacity_ratio.iloc[:, 0:n_runs].divide(df_plant_capacity_ratio['MATPOWER Capacity (MW)'], axis='index') # Hardcoded

    # Combine with input factors
    df_plant_capacity_ratio = df_plant_capacity_ratio.transpose()
    df_plant_capacity_ratio = df_plant_capacity_ratio.join(df[uniform_factor_labs+obj_labs])

    # Get Cooling System Type
    df_plant_capacity_ratio = pd.melt(df_plant_capacity_ratio, value_vars=plant_names,
                                      id_vars=uniform_factor_labs + obj_labs, var_name='Plant Name', value_name='Output')
    df_plant_capacity_ratio = df_plant_capacity_ratio.merge(
        df_gen_info[['Plant Name', '923 Cooling Type', 'MATPOWER Fuel']])

    return df_plant_capacity_ratio


def viz_effect_of_withdrawal_weight_plant_output(df, df_gen_info):

    # Subsetting data
    df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
    withdrawal_criteria = [0.0, 0.1, 0.1]
    uniform_water_criteria = [0.5, 0.5, 1.5]
    case_a = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[0]) & (df['Uniform Water Coefficient'] == uniform_water_criteria[0])
    case_b = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[1]) & (df['Uniform Water Coefficient'] == uniform_water_criteria[1])
    case_c = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[2]) & (df['Uniform Water Coefficient'] == uniform_water_criteria[2])
    df = df[case_a | case_b | case_c]


    # Making labels
    df['Withdrawal Weight ($/Gallon)'] = df['Withdrawal Weight ($/Gallon)'].astype('category')
    df['Fuel/Cooling Type'] = df['MATPOWER Fuel'] + '/' + df['923 Cooling Type']
    df = df.sort_values('Fuel/Cooling Type')
    df['ID'] = '$w_{with}=$' + df['Withdrawal Weight ($/Gallon)'].astype(str) + ', $c_{water}=$' + df['Uniform Water Coefficient'].astype(str)

    # Creating plot
    g = sns.FacetGrid(df, row='ID', col='Fuel/Cooling Type', sharex='col', aspect=0.8, margin_titles=True)
    g.map_dataframe(sns.lineplot, y='Output', x='Uniform Loading Coefficient', hue='Plant Name', style='Plant Name',
                    markers=True)
    for row in range(g.axes.shape[0]):
        for col in range(g.axes.shape[1]):
            if row == 0:
                g.axes[row, col].set_title(df['Fuel/Cooling Type'].unique()[col])
            else:
                g.axes[row, col].set_title('')
            if col == g.axes.shape[1] - 1:
                txt = 'Uniform Water Coefficient = ' + str(uniform_water_criteria[row]) + \
                      '\nWithdrawal Weight ($/Gallon) = ' + str(withdrawal_criteria[row])
                g.axes[row, col].texts[0].set_text(txt)
                g.axes[row, col].texts[0].set_fontsize(10)
            if row == 2:
                g.axes[row, col].legend(loc='center', bbox_to_anchor=(0.5, -0.6))
    g.set_axis_labels(x_var='Uniform Loading Coefficient', y_var='Capacity Ratio')
    plt.tight_layout()
    plt.show()

    return g


def uniform_sa_dataviz(df, uniform_factor_labs, obj_labs, df_gen_info_match_water):
    fig_a = viz_effect_of_withdrawal_weight(df, uniform_factor_labs, obj_labs)
    df_plant_capacity_ratio = get_plant_output_ratio(df, df_gen_info_match_water, uniform_factor_labs, obj_labs)
    fig_b = viz_effect_of_withdrawal_weight_plant_output(df_plant_capacity_ratio, df_gen_info_match_water)
    return fig_a, fig_b


def fitSingleModels(df, obj_labs, factor_labs):
    mods = {}
    for i in obj_labs:
        clf = tree.DecisionTreeRegressor(random_state=1008, max_depth=5, max_leaf_nodes=12)
        mods[i] = clf.fit(X=df[factor_labs], y=df[i])
    return mods


def dtreeViz(mods, df, uniform_factor_labs):
    drawing_ls = []
    for key in mods:
        viz = dtreeviz(mods[key],
                       df[uniform_factor_labs],
                       df[key],
                       target_name=key,
                       feature_names=uniform_factor_labs,
                       orientation='LR')
        viz.save('temp.svg')
        drawing_ls.append(svg2rlg('temp.svg'))
        os.remove('temp.svg')
        os.remove('temp')
    return drawing_ls


def uniform_sa_tree(df, obj_labs, uniform_factor_labs):
    mods = fitSingleModels(df, obj_labs, uniform_factor_labs)
    drawing_ls = dtreeViz(mods, df, uniform_factor_labs)
    return drawing_ls


def generate_nonuniform_samples(n_sample, df_gen_info_match_water, df_hnwc):

    # Add plant information
    df_hnwc = df_gen_info_match_water[['Plant Name', '923 Cooling Type', 'MATPOWER Fuel']].merge(df_hnwc)
    df_hnwc['Input Factor'] = df_hnwc['Plant Name'] + ' Non-Uniform Water Coefficient'

    # Sampling
    replace = True  # with replacement
    n_inputs_factors = len(df_hnwc['Input Factor'].unique())
    fn = lambda obj: obj.loc[np.random.choice(obj.index, n_sample, replace), :]
    df_sample = df_hnwc.groupby('Input Factor', as_index=False).apply(fn)
    df_sample['Sample Index'] = np.tile(np.arange(0, n_sample), n_inputs_factors)
    df_sample = df_sample.reset_index().pivot(columns='Input Factor', values='value', index='Sample Index')
    return df_sample, df_hnwc


def nonuniform_factor_multiply(ser_c_water, c_load, exogenous_labs, net, df_geninfo):

    # Nonuniform coefficients
    df_geninfo = df_geninfo.merge(ser_c_water, left_on=['Plant Name'], right_index=True, how='left')
    df_geninfo.iloc[:, -1].fillna(0, inplace=True)  # Joined column will always be last

    # Uniform coefficients
    beta_load = net.load['p_mw'].values * c_load
    beta_with = (df_geninfo['Median Withdrawal Rate (Gallon/kWh)'] * df_geninfo.iloc[:,-1]).values
    beta_con = (df_geninfo['Median Consumption Rate (Gallon/kWh)'] * df_geninfo.iloc[:,-1]).values
    vals = np.concatenate((beta_with, beta_con, beta_load))
    idxs = exogenous_labs[2:]
    return pd.Series(vals, index=idxs)


def get_nonuniform_exogenous(df_sample, df_hnwc, w_with, w_con, c_load, df_geninfo, net, exogenous_labs):
    # Formatting
    original_columns = df_sample.columns
    df_sample.columns = [i.split(' Non-Uniform Water Coefficient')[:][0] for i in df_sample.columns]

    # Multiply exogenous parameters
    df_exogenous = df_sample.apply(lambda row: nonuniform_factor_multiply(row, c_load, exogenous_labs, net, df_geninfo), axis=1)

    # Storing
    df_sample.columns = original_columns
    df_sample['Withdrawal Weight ($/Gallon)'] = w_with
    df_sample['Consumption Weight ($/Gallon)'] = w_con
    df_sample['Uniform Loading Coefficient'] = c_load
    df_exogenous = pd.concat([df_sample, df_exogenous], axis=1)
    return df_exogenous


def MGSA_FirstOrder(Input, Output, ndomain):
    '''
    input: nsample * nd matrix, where nsample is the number of sample, and nd
    is the input dimension
    output: nsample * 1 array
    ndomain: number of sub-domain the to divide a single input
    This algorithm is proposed by me. Please cite the following paper if you use this code.
    Li, Chenzhao, and Sankaran Mahadevan. "An efficient modularized sample-based method to estimate the first-order Sobol’index." Reliability Engineering & System Safety (2016).

    TODO before publishing, make sure this is just cloned from his site
    '''
    (nsample, nd) = np.shape(Input);

    # convert the input samples into cdf domains
    U = np.linspace(0.0, 1.0, num=ndomain + 1)
    cdf_input = np.zeros((nsample, nd))
    cdf_values = np.linspace(1.0 / nsample, 1.0, nsample)

    j = 1
    for i in range(nd):
        IX = np.argsort(Input[:, i])
        IX2 = np.argsort(IX)
        cdf_input[:, i] = cdf_values[IX2]

    # compute the first-order indices
    VY = np.var(Output, ddof=1)
    VarY_local = np.zeros((ndomain, nd))
    for i in range(nd):
        cdf_input_i = cdf_input[:, i]
        output_i = Output
        U_i = U
        for j in range(ndomain):
            sub = cdf_input_i < U_i[j + 1]
            VarY_local[j, i] = np.var(output_i[sub], ddof=1)
            inverse_sub = ~sub
            cdf_input_i = cdf_input_i[inverse_sub]
            output_i = output_i[inverse_sub]

    index = 1.0 - np.mean(VarY_local, axis=0) / VY

    return index


def nonuniform_sa(df_gen_info, df_hnwc, df_operation, obj_labs, n_tasks, net):
    # Internal Varrs
    exogenous_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)'] + \
                     ('MATPOWER Generator ' + df_gen_info['MATPOWER Index'].astype(str) + ' Withdrawal Rate (Gallon/kWh)').tolist() + \
                     ('MATPOWER Generator ' + df_gen_info['MATPOWER Index'].astype(str) + ' Consumption Rate (Gallon/kWh)').tolist() + \
                     ('PANDAPOWER Bus ' + net.load['bus'].astype(str) + ' Load (MW)').tolist()
    results_labs = obj_labs + ('MATPOWER Generator ' + df_gen_info['MATPOWER Index'].astype(str) + ' Ratio of Capacity').to_list()
    t = 5 * 1 / 60 * 1000  # minutes * hr/minutes * kw/MW
    n_sample = 1024 * (2*10+2)  # for saltelli sampling 1024
    print('Success: Initialized Non-Uniform')
    results_ls = []
    sobol_ls = []

    # Adding Pandapower information
    df_gen_info = matchIndex(df_gen_info, net)

    # Generate Samples
    df_sample, df_hnwc = generate_nonuniform_samples(n_sample, df_gen_info, df_hnwc)
    factor_labs = df_sample.columns.to_list()
    print('Number of Samples: ', len(df_sample))

    for index, row in df_operation.iterrows():
        # Apply Coefficients to Exogenous Parameters
        df_exogenous = get_nonuniform_exogenous(
            df_sample.copy(),
            df_hnwc,
            row['Withdrawal Weight ($/Gallon)'],
            row['Consumption Weight ($/Gallon)'],
            row['Uniform Loading Factor'],
            df_gen_info,
            net,
            exogenous_labs
        )

        # Evaluate model
        ddf_exogenous = dd.from_pandas(df_exogenous, npartitions=n_tasks)
        df_results = ddf_exogenous.apply(
            lambda row: waterOPF(
                row[exogenous_labs],
                t,
                results_labs,
                net,
                df_gen_info
            ),
            axis=1,
            meta={key: 'float64' for key in results_labs}
        ).compute(scheduler='processes')
        df_results_exogenous = pd.concat([df_results, df_exogenous], axis=1)
        df_results_exogenous = df_results_exogenous.dropna()
        df_results_exogenous = df_results_exogenous.drop_duplicates()  # Sometimes dask duplicates rows
        print('Success: ' + row['Operational Scenario'] + ' Model Run Complete')

        # Calculate sobol
        df_sobol_results = pd.DataFrame()
        for i in obj_labs:
            ndomain = int(np.sqrt(n_sample))
            si_vals = MGSA_FirstOrder(Input=df_results_exogenous[factor_labs].values,
                                      Output=df_results_exogenous[i].values,
                                      ndomain=ndomain)
            df_sobol_results = df_sobol_results.append(pd.Series(si_vals, index=factor_labs).rename(i))
        print('Success: ' + row['Operational Scenario'] + ' Sobol Analysis')

        # Storage
        df_results_exogenous['Operational Scenario'] = row['Operational Scenario']
        df_sobol_results['Operational Scenario'] = row['Operational Scenario']
        sobol_ls.append(df_sobol_results.rename_axis('Objective').reset_index())
        results_ls.append(df_results_exogenous)

    # Creating main dataframes
    df_nonuniform = pd.concat(results_ls, ignore_index=True)
    df_nonuniform_sobol = pd.concat(sobol_ls, ignore_index=True)

    return df_nonuniform, df_nonuniform_sobol


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    data = data.pivot(columns='Plant Name', index='Objective', values='Sobol Index')
    return sns.heatmap(data, **kwargs)


def nonuniform_sobol_viz(df_sobol, df_gen_info):

    # Local variables
    input_factor_labs = df_sobol.filter(like='Non-Uniform Water Coefficient').columns

    # Filter
    invalid_index = (df_sobol['Operational Scenario'].str.contains('OPF')) & \
                    (df_sobol['Objective'].str.contains('Cost'))  # Objectives that have no impact
    df_sobol.loc[invalid_index, input_factor_labs] = np.nan
    df_sobol._get_numeric_data()[df_sobol._get_numeric_data() < 0] = 0  # Due to numeric estimation

    # Get plant information
    df_sobol = df_sobol.melt(
        id_vars=['Objective', 'Operational Scenario'],
        value_vars=input_factor_labs,
        var_name='Input Factor',
        value_name='Sobol Index'
    )
    df_sobol['Plant Name'] = df_sobol['Input Factor'].str.split(' Non-Uniform Water Coefficient', expand=True)[0]
    df_plant_info = df_gen_info.groupby('Plant Name').first().reset_index()
    df_plant_info['Fuel/Cooling Type'] = df_plant_info['MATPOWER Fuel'] + '/' + df_plant_info['923 Cooling Type']
    df_sobol = df_sobol.merge(df_plant_info[['Plant Name', 'Fuel/Cooling Type']])

    # Plot
    min_sobol = df_sobol['Sobol Index'].min()
    max_sobol = df_sobol['Sobol Index'].max()
    g = sns.FacetGrid(df_sobol, col='Fuel/Cooling Type', row='Operational Scenario', sharex='col', sharey=True)
    cbar_ax = g.fig.add_axes([.87, .15, .03, .7])
    g.map_dataframe(draw_heatmap, cmap='viridis', cbar_ax=cbar_ax, cbar_kws={'label': 'First Order Sobol Index Value'}, vmin=min_sobol, vmax=max_sobol)
    g.set_titles(rotation=5)
    g.fig.subplots_adjust(top=0.9, right=0.75)
    plt.show()


    return g


def historic_load_viz(df):

    # Formatting
    df = df.rename({'ActualLoad': 'Actual Load (MW)'}, axis='columns')
    df['Uniform Loading Coefficient'] = df['Actual Load (MW)'] / df['Actual Load (MW)'].median()

    # Plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    sns.histplot(data=df, x='Actual Load (MW)', ax=ax1)
    sns.histplot(data=df, x='Uniform Loading Coefficient', ax=ax2)
    plt.tight_layout()
    plt.show()
    return fig


def get_system_information(df_gen_info_match_water):

    # Generator aggregation
    df_gens = df_gen_info_match_water.groupby('Plant Name')['MATPOWER Index'].apply(set).reset_index(name='Generators')
    df_gens['Generators'] = df_gens.apply(
        lambda row: ', '.join([str(element) for element in row['Generators']]),
        axis=1
    )

    df_capacity = df_gen_info_match_water.groupby('Plant Name').sum()

    # Remaining information
    df_info = df_gen_info_match_water.groupby('Plant Name').first().drop('MATPOWER Capacity (MW)', axis='columns')

    # Merging
    df_info = df_info.join(df_capacity['MATPOWER Capacity (MW)'])
    df = df_info.merge(df_gens, left_index=True, right_on='Plant Name')

    return df
