import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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
