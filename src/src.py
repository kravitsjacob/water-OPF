import os
import copy

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import dask.dataframe as dd
import pandapower as pp
import itertools


def grid_setup(net):
    # Dispatching all generators
    net.sgen['in_service'] = True
    net.gen['in_service'] = True
    net.ext_grid['in_service'] = True

    return net


def generator_match(df_gen_info):
    # This is done manually through remotely sources images and spatial analysis
    manual_dict = {0: {'MATPOWER Index': 49,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         1: {'MATPOWER Index': 50,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         2: {'MATPOWER Index': 51,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         3: {'MATPOWER Index': 52,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         4: {'MATPOWER Index': 53,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         5: {'MATPOWER Index': 65,
             'EIA Plant Name': 'Pioneer Trail Wind Farm, LLC',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'PAXTON 1'},
         6: {'MATPOWER Index': 67,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         7: {'MATPOWER Index': 68,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         8: {'MATPOWER Index': 69,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         9: {'MATPOWER Index': 70,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         10: {'MATPOWER Index': 71,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         11: {'MATPOWER Index': 72,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         12: {'MATPOWER Index': 73,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         13: {'MATPOWER Index': 76,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         14: {'MATPOWER Index': 77,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         15: {'MATPOWER Index': 78,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         16: {'MATPOWER Index': 79,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         17: {'MATPOWER Index': 90,
              'EIA Plant Name': 'Clinton LFGTE',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 3'},
         18: {'MATPOWER Index': 91,
              'EIA Plant Name': 'Clinton LFGTE',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 3'},
         19: {'MATPOWER Index': 92,
              'EIA Plant Name': 'Clinton LFGTE',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 3'},
         20: {'MATPOWER Index': 94,
              'EIA Plant Name': 'Tuscola Station',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'TUSCOLA 2'},
         21: {'MATPOWER Index': 104,
              'EIA Plant Name': 'High Trail Wind Farm LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'ELLSWORTH 1'},
         22: {'MATPOWER Index': 105,
              'EIA Plant Name': 'High Trail Wind Farm LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'ELLSWORTH 1'},
         23: {'MATPOWER Index': 114,
              'EIA Plant Name': 'White Oak Energy LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'NORMAL 2'},
         24: {'MATPOWER Index': 115,
              'EIA Plant Name': 'White Oak Energy LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'NORMAL 2'},
         25: {'MATPOWER Index': 125,
              'EIA Plant Name': 'E D Edwards',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'BARTONVILLE'},
         26: {'MATPOWER Index': 126,
              'EIA Plant Name': 'E D Edwards',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'BARTONVILLE'},
         27: {'MATPOWER Index': 127,
              'EIA Plant Name': 'E D Edwards',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'BARTONVILLE'},
         28: {'MATPOWER Index': 135,
              'EIA Plant Name': 'Powerton',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'PEKIN 1'},
         29: {'MATPOWER Index': 136,
              'EIA Plant Name': 'Powerton',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'PEKIN 1'},
         30: {'MATPOWER Index': 147,
              'EIA Plant Name': 'Rail Splitter Wind Farm',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'HOPEDALE 2'},
         31: {'MATPOWER Index': 151,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         32: {'MATPOWER Index': 152,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         33: {'MATPOWER Index': 153,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         34: {'MATPOWER Index': 154,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         35: {'MATPOWER Index': 155,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         36: {'MATPOWER Index': 161,
              'EIA Plant Name': 'Interstate',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 4'},
         37: {'MATPOWER Index': 164,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         38: {'MATPOWER Index': 165,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         39: {'MATPOWER Index': 166,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         40: {'MATPOWER Index': 167,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         41: {'MATPOWER Index': 168,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         42: {'MATPOWER Index': 169,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         43: {'MATPOWER Index': 170,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         44: {'MATPOWER Index': 182,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 2'},
         45: {'MATPOWER Index': 183,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 2'},
         46: {'MATPOWER Index': 189,
              'EIA Plant Name': 'Clinton Power Station',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 1'},
         47: {'MATPOWER Index': 196,
              'EIA Plant Name': 'Gibson City Energy Center LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'GIBSON CITY 1'},
         48: {'MATPOWER Index': 197,
              'EIA Plant Name': 'Gibson City Energy Center LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'GIBSON CITY 1'}}
    df_matches = pd.DataFrame.from_records(manual_dict).T

    # Merge manual matches
    df_gen_info_match = df_gen_info.merge(df_matches)
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


def get_historic_nonuniform_water_coefficients(df_region):
    df_region['Uniform Water Coefficient Consumption'] = df_region['Withdrawal Rate (Gallon/kWh)'] / df_region['Withdrawal Rate (Gallon/kWh) Median']
    df_region['Uniform Water Coefficient Withdrawal'] = df_region['Consumption Rate (Gallon/kWh)'] / df_region['Consumption Rate (Gallon/kWh) Median']
    df = df_region.melt(value_vars=['Uniform Water Coefficient Consumption', 'Uniform Water Coefficient Withdrawal'],
                      id_vars=['923 Cooling Type', 'Fuel Type'], var_name='Exogenous Parameter')
    df['Input Factor'] = 'C water ' + df['Fuel Type'] + ' ' + df['923 Cooling Type']
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


def hnwc_plotter(df):
    g = sns.FacetGrid(df, col='Input Factor', col_wrap=2, sharex=False, sharey=False, aspect=1.5)
    g.map(sns.histplot, 'value', stat='density')
    g.axes[0].set_title('$C_{water,coal,induced-recirculating}$')
    g.axes[1].set_title('$C_{water,coal,recirculating}$')
    g.axes[2].set_title('$C_{water,coal,once-through}$')
    g.axes[3].set_title('$C_{water,natural-gas,induced-recirculating}$')
    g.axes[4].set_title('$C_{water,nuclear,recirculating}$')
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
    df_hnwc = get_historic_nonuniform_water_coefficients(df_region)

    # Plotting
    fig_regionDistribututionPlotter = regionDistribututionPlotter(df_region).figure
    fig_regionBoxPlotter = regionBoxPlotter(df_region).fig
    fig_coalPlotter = coalPlotter(df_region).fig
    fig_hnwc_plotter = hnwc_plotter(df_hnwc).fig

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
    c_load_vals = np.linspace(df_gridspecs['Min']['Uniform Loading Factor'],
                              df_gridspecs['Max']['Uniform Loading Factor'],
                              df_gridspecs['Number of Steps']['Uniform Loading Factor'])
    c_water_vals = np.linspace(df_gridspecs['Min']['Uniform Water Factor'],
                               df_gridspecs['Max']['Uniform Water Factor'],
                               df_gridspecs['Number of Steps']['Uniform Water Factor'])
    # Create Grid
    df_search = pd.DataFrame(list(itertools.product(w_with_vals, w_con_vals, c_load_vals, c_water_vals)), columns=df_gridspecs.index)
    # Multiply exogenous parameters
    beta_with = df_geninfo['Median Withdrawal Rate (Gallon/kWh)'].values
    beta_con = df_geninfo['Median Consumption Rate (Gallon/kWh)'].values
    beta_load = net.load['p_mw'].values
    df_exogenous = df_search.apply(lambda row: uniformFactorMultiply(row['Uniform Water Factor'], row['Uniform Loading Factor'], beta_with, beta_con, beta_load, exogenous_labs), axis=1)
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
    if state is 'converge':
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
    elif state is 'not converge':
        F_cos = F_with = F_con = F_gen = np.nan
        internal_decs = [np.nan]*len(df_geninfo)
    return pd.Series([F_cos, F_gen, F_with, F_con] + internal_decs, index=results_labs)


def uniform_sa(df_gen_info_match_water, net, n_tasks):
    # Initialize
    exogenous_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)'] + \
                     ('MATPOWER Generator ' + df_gen_info_match_water['MATPOWER Index'].astype(str) + ' Withdrawal Rate (Gallon/kWh)').tolist() + \
                     ('MATPOWER Generator ' + df_gen_info_match_water['MATPOWER Index'].astype(str) + ' Consumption Rate (Gallon/kWh)').tolist() + \
                     ('PANDAPOWER Bus ' + net.load['bus'].astype(str) + ' Load (MW)').tolist()
    inputfactor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Loading Factor', 'Uniform Water Factor']
    results_labs = ['Total Cost ($)', 'Generator Cost ($)', 'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)'] + \
                   ('MATPOWER Generator ' + df_gen_info_match_water['MATPOWER Index'].astype(str) + ' Ratio of Capacity').to_list()
    df_gridspecs = pd.DataFrame(data=[[0.0, 0.1, 10], [0.0, 1.0, 10], [1.0, 1.5, 10], [0.5, 1.5, 10]],
                                index=inputfactor_labs, columns=['Min', 'Max', 'Number of Steps'])
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