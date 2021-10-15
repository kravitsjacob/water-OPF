import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def uniform_water_coefficient_distribution(df):
    data = (df['Withdrawal Rate (Gallon/kWh)'] / df['Withdrawal Rate (Gallon/kWh) Median']).append(
        df['Consumption Rate (Gallon/kWh)'] / df['Consumption Rate (Gallon/kWh) Median'])
    g = sns.histplot(data)
    plt.show()
    return g


def region_water_boxplots(df):
    df_plot = df.melt(value_vars=['Withdrawal Rate (Gallon/kWh)', 'Consumption Rate (Gallon/kWh)'],
                      id_vars=['923 Cooling Type', 'Fuel Type'], var_name='Exogenous Parameter')
    g = sns.catplot(
        data=df_plot,
        x='value',
        y='923 Cooling Type',
        col='Exogenous Parameter',
        row='Fuel Type',
        kind='box',
        sharex='col',
        height=3,
        aspect=1.2
    )
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


def coal_scatter_kmeans(df):
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


def hnwc_histograms(df, df_gen_info):
    # Titles
    df_titles = df_gen_info.sort_values('MATPOWER Fuel')
    df_titles['Fuel/Cooling Type'] = df_titles['MATPOWER Fuel'] + '/' + df_titles['923 Cooling Type']
    df_titles = df_titles.groupby('Fuel/Cooling Type')['Plant Name'].apply(set).reset_index(name='Plants')
    df_titles['Grouped Plants'] = df_titles.apply(
        lambda row: 'Plant ID: ' + ", ".join(
            [item for subitem in row['Plants'] for item in subitem.split() if item.isdigit()]
        ),
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
