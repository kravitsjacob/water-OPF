import os

import numpy as np
import pandapower.plotting as ppp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from svglib.svglib import svg2rlg
import dtreeviz.trees as dtviz  # Requires pip version, conflicts with pandapower.plotting
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


def effect_of_withdrawal_weight_on_withdrawal(df, uniform_factor_labs, obj_labs):
    # Subsetting data
    plot_df = df[uniform_factor_labs + obj_labs]
    plot_df = plot_df[plot_df['Consumption Weight ($/Gallon)'] == 0.0]
    plot_df = plot_df[plot_df['Withdrawal Weight ($/Gallon)'].isin([0.0, 0.1])]
    plot_df = plot_df.round({'Uniform Water Coefficient': 2})
    plot_df = plot_df[plot_df['Uniform Water Coefficient'].isin(
        plot_df['Uniform Water Coefficient'].unique()[1::2])]  # Select every other uniform water coefficient

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=plot_df,
        x='Uniform Loading Coefficient',
        y='Water Withdrawal (Gallon)',
        hue='Uniform Water Coefficient',
        style='Withdrawal Weight ($/Gallon)',
        markers=True,
        ax=ax
    )
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return fig


def effect_of_withdrawal_weight_plant_output(df):
    # Subsetting data
    df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
    withdrawal_criteria = [0.0, 0.1, 0.1]
    uniform_water_criteria = [0.5, 0.5, 1.5]
    case_a = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[0]) &\
             (df['Uniform Water Coefficient'] == uniform_water_criteria[0])
    case_b = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[1]) &\
             (df['Uniform Water Coefficient'] == uniform_water_criteria[1])
    case_c = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[2]) &\
             (df['Uniform Water Coefficient'] == uniform_water_criteria[2])
    df = df[case_a | case_b | case_c]

    # Making labels
    df['Withdrawal Weight ($/Gallon)'] = df['Withdrawal Weight ($/Gallon)'].astype('category')
    df['Fuel/Cooling Type'] = df['MATPOWER Fuel'] + '/' + df['923 Cooling Type']
    df = df.sort_values(['Uniform Water Coefficient', 'Fuel/Cooling Type'])
    df['ID'] = '$w_{with}=$' + df['Withdrawal Weight ($/Gallon)'].astype(str) +\
               ', $c_{water}=$' + df['Uniform Water Coefficient'].astype(str)

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


def table_effect_of_withdrawal_weight_line_flows(df):
    # Subsetting data
    df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
    df = df[df['Uniform Loading Coefficient'] == 1.5]
    df = df[df['Uniform Water Coefficient'] == 0.5]
    withdrawal_criteria = [0.0, 0.1]
    case_a = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[0])
    case_b = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[1])
    df = df[case_a | case_b]

    # Making labels
    df['ID'] = 'Withdrawal Weight ($/Gallon) = ' + df['Withdrawal Weight ($/Gallon)'].astype(str)

    # Formatting data
    line_labs = df.filter(like='Loading (Percent)').columns
    df = df.set_index('ID')
    df = df[line_labs].transpose()
    df = df.reset_index().rename(columns={'index': 'ID'})
    df['Line'] = df['ID'].str.split(' Loading', expand=True)[0]

    # Getting absolute difference
    df['Absolute Difference (Percent)'] = abs(
        df['Withdrawal Weight ($/Gallon) = 0.0'] - df['Withdrawal Weight ($/Gallon) = 0.1']
    )
    df = df.sort_values('Absolute Difference (Percent)', ascending=False)

    return df


def effect_of_withdrawal_weight_line_flows(net_diff):
    # Visualize flow differences
    net_diff.res_line['loading_percent'] = net_diff.res_line['Change in Loading (Percent)']
    net_diff.res_trafo['loading_percent'] = net_diff.res_trafo['Change in Loading (Percent)']
    ppp.create_generic_coordinates(net_diff, overwrite=True)
    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-55.0, vmax=55.0)
    pc = ppp.create_bus_collection(net_diff, size=0.1, alpha=0.2)
    lc = ppp.create_line_collection(
        net_diff, use_bus_geodata=True, cmap=cmap, norm=norm, cbar_title='Change in Line Loading (%)'
    )
    _, tlc = ppp.create_trafo_collection(net_diff, cmap=cmap, norm=norm, size=0.05)
    tpc, _ = ppp.create_trafo_collection(net_diff, size=0.05, alpha=0.2)
    ppp.draw_collections([tpc, tlc, pc, lc])
    fig = plt.gcf()
    plt.show()

    return fig


def decision_tree(mods, obj_lab, df, uniform_factor_labs):
    try:
        viz = dtviz.dtreeviz(
            mods[obj_lab],
            df[uniform_factor_labs],
            df[obj_lab],
            target_name=obj_lab,
            feature_names=uniform_factor_labs,
            orientation='LR',
            precision=0
        )
        viz.save('temp.svg')
        print(f'Success: Tree created for {obj_lab}')
        svg_fig = svg2rlg('temp.svg')
        os.remove('temp.svg')
        os.remove('temp')
        return svg_fig
    except AttributeError:
        print('AttributeError: Cannot use both `dtreeviz` and `pandapower.plotting`')
        return 1


def draw_heatmap(**kwargs):
    data = kwargs.pop('data')
    data = data.pivot(columns='Plant Name', index='Objective', values='Sobol Index')
    return sns.heatmap(data, **kwargs)


def nonuniform_sobol_heatmap(df_sobol, df_gen_info):
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
    df_sobol = df_sobol.sort_values('Fuel/Cooling Type')

    # Plot
    min_sobol = df_sobol['Sobol Index'].min()
    max_sobol = df_sobol['Sobol Index'].max()
    g = sns.FacetGrid(
        df_sobol,
        col='Fuel/Cooling Type',
        row='Operational Scenario',
        sharex='col',
        sharey=True,
        aspect=1.0,
        height=1.7,
        margin_titles=True
    )
    cbar_ax = g.fig.add_axes([.87, .15, .03, .7])
    g.map_dataframe(
        draw_heatmap,
        cmap='viridis',
        cbar_ax=cbar_ax,
        cbar_kws={'label': 'First Order Sobol Index Value'},
        vmin=min_sobol,
        vmax=max_sobol
    )
    g.fig.subplots_adjust(left=0.28, right=0.75, top=0.85, bottom=0.12)
    for row in range(g.axes.shape[0]):
        for col in range(g.axes.shape[1]):
            if row == 0:
                g.axes[row, col].set_title(df_sobol['Fuel/Cooling Type'].unique()[col], rotation=90)
            else:
                g.axes[row, col].set_title('')

            if col == g.axes.shape[1] - 1:
                txt = df_sobol['Operational Scenario'].unique()[row].replace(' ', '\n')
                g.axes[row, col].texts[0].set_text(txt)
                g.axes[row, col].texts[0].set(multialignment='center', x=1.8, ha='center', rotation='horizontal')

    g.set_xticklabels(rotation=90)
    g.set_xlabels('')
    g.set_ylabels('')

    # Display plot
    plt.show()

    return g


def historic_load_hist(df):
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


def system_information(df_gen_info):
    # Generator aggregation
    df_gens = df_gen_info.groupby('Plant Name')['MATPOWER Index'].apply(set).reset_index(name='Generators')
    df_gens['Generators'] = df_gens.apply(
        lambda row: ', '.join([str(element) for element in row['Generators']]),
        axis=1
    )

    df_capacity = df_gen_info.groupby('Plant Name').sum()

    # Remaining information
    df_info = df_gen_info.groupby('Plant Name').first().drop('MATPOWER Capacity (MW)', axis='columns')

    # Merging
    df_info = df_info.join(df_capacity['MATPOWER Capacity (MW)'])
    df = df_info.merge(df_gens, left_index=True, right_on='Plant Name')

    return df
