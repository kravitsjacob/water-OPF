
# Import Packages
import pandas as pd
import os
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats


# Global Vars
pathto_EIA = 'G:\My Drive\Documents (Stored)\data_sets\EIA_theremoelectric_water_use'
pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.1'
pathto_geninfo = os.path.join(pathto_data, 'synthetic_grid', 'gen_info_match.csv')
pathto_h5 = os.path.join(pathto_data, 'temp', 'processing_data.h5')
pathto_results = os.path.join(pathto_data, 'output', 'gen_info_match_water.csv')



def importEIAData():
    # Local Vars
    years = ['2019', '2018', '2017', '2016', '2015', '2014']
    df_list = []
    # Import all dataframes
    for i in years:
        path = os.path.join(pathto_EIA, 'cooling_detail_'+i+'.xlsx')
        print(path)
        # Import Dataframe
        df_temp = pd.read_excel(path, header=2)
        # Replace space values with nan values
        df_temp = df_temp.replace(r'^\s*$', np.nan, regex=True)
        df_list.append(df_temp)
    # Concat Dataframes into Single Dataframe
    df = pd.concat(df_list)
    # Store in HDF5 Format
    s = pd.HDFStore(pathto_h5)
    s.put('importData_df', df)
    s.close()
    return 0


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


def regionDistribututionPlotter(df):
    df['Uniform Water Coefficient'] = df['Withdrawal Rate (Gallon/kWh)'] / df['Withdrawal Rate (Gallon/kWh) Median']
    fig, ax1 = plt.subplots()
    sns.histplot(data=df, x='Uniform Water Coefficient', ax=ax1, bins=20)
    plt.tight_layout()
    plt.show()

    df['Uniform Water Coefficient'] = df['Consumption Rate (Gallon/kWh)'] / df['Consumption Rate (Gallon/kWh) Median']
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='Uniform Water Coefficient', ax=ax2, bins=20)
    plt.tight_layout()
    plt.show()

    return fig, fig2


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


def regionFitandHistPlotter(df_region, df_fitted):
    df_region['Uniform Water Coefficient Consumption'] = df_region['Withdrawal Rate (Gallon/kWh)'] / df_region['Withdrawal Rate (Gallon/kWh) Median']
    df_region['Uniform Water Coefficient Withdrawal'] = df_region['Consumption Rate (Gallon/kWh)'] / df_region['Consumption Rate (Gallon/kWh) Median']
    df_plot = df_region.melt(value_vars=['Uniform Water Coefficient Consumption', 'Uniform Water Coefficient Withdrawal'],
                      id_vars=['923 Cooling Type', 'Fuel Type'], var_name='Exogenous Parameter')
    df_plot['Cooling System/Fuel Type'] = df_plot['923 Cooling Type'] + '/' + df_plot['Fuel Type']
    # Fit
    fig, axes = plt.subplots(len(df_plot['Cooling System/Fuel Type'].unique()), figsize=(8,15))
    for i, j in enumerate(df_plot['Cooling System/Fuel Type'].unique()):
        samp = df_plot[df_plot['Cooling System/Fuel Type'] == j]['value'].values
        loc = 0.0
        shape = df_fitted['Sigma'][j]
        scale = np.exp(df_fitted['Mu'][j])
        ax = sns.histplot(samp, kde=False, stat='density', label='Historic', ax=axes[i])
        x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
        x_pdf = np.linspace(x0, x1, 100)
        y_pdf = stats.lognorm(shape, loc, scale).pdf(x_pdf)
        axes[i].plot(x_pdf, y_pdf, 'r', lw=2, label='pdf')
        ax.legend()
        ax.set_title(j)
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    return fig


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


def main():
    # Import
    df_geninfo = pd.read_csv(pathto_geninfo)
    #importEIAData()
    df_EIA = pd.read_hdf(pathto_h5, 'importData_df')
    # Get Cooling System From Matched EIA Generators
    df_geninfo = getCoolingSystem(df_EIA, df_geninfo)
    # Get Regional Estimates of Water Use
    df_region = getRegional(df_EIA)
    # Get Generator Water Use
    df_geninfo, df_region = getGeneratorWaterData(df_region, df_geninfo)
    # Median Case
    df_geninfo['Median Withdrawal Rate (Gallon/kWh)'] = df_geninfo.apply(lambda row: np.median(row['Withdrawal Rate Cluster Data (Gallon/kWh)']), axis=1)
    df_geninfo['Median Consumption Rate (Gallon/kWh)'] = df_geninfo.apply(lambda row: np.median(row['Consumption Rate Cluster Data (Gallon/kWh)']), axis=1)

    # Plotting
    fig1, fig2 = regionDistribututionPlotter(df_region)
    fig1.savefig('../figures/uniform water coefficient distribution (withdrawal).pdf')
    fig2.savefig('../figures/uniform water coefficient distribution (consumption).pdf')
    regionBoxPlotter(df_region).fig.savefig('../figures/region water boxplots.pdf')
    # regionFitandHistPlotter(df_region, df_fitted).savefig('Uniform Water Coefficient Fitted Distributions.pdf')
    coalPlotter(df_region).fig.savefig('../figures/coal scatter kmeans.pdf')
    # # Export
    # df_geninfo.to_csv(pathto_results, index=False)
    return 0


if __name__ == '__main__':
    main()
