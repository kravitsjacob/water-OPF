
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2'
pathto_samples = os.path.join(pathto_data, 'uniform_sa_samples', 'samples.csv')
pathto_gen_info = os.path.join(pathto_data, 'synthetic_grid', 'gen_info_match_water.csv')
pathto_figures = os.path.join(pathto_data, 'figures')

factor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor']
obj_labs = ['Total Cost ($)', 'Generator Cost ($)',	'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)']


def viz_effect_of_withdrawal_weight(df):
    # Prepare Data
    plot_df = df[factor_labs + obj_labs]
    plot_df = plot_df[plot_df['Consumption Weight ($/Gallon)'] == 0.0]
    plot_df = plot_df.round({'Uniform Water Factor': 2})
    plot_df = plot_df[plot_df['Uniform Water Factor'].isin(plot_df['Uniform Water Factor'].unique()[1::2])]  # Select every other loading factor value
    plot_df['Withdrawal Weight ($/Gallon)'] = pd.cut(plot_df['Withdrawal Weight ($/Gallon)'], [0.0, 0.01, 0.1], labels=['0.0', '[0.01, 0.1]'], right=False)
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    g = sns.lineplot(data=plot_df, x='Uniform Loading Factor', y='Water Withdrawal (Gallon)',
                     hue='Uniform Water Factor', style='Withdrawal Weight ($/Gallon)', markers=True, ax=ax)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return fig


def get_fuelcool_output_ratio(df, df_gen_info):
    # Internal vars
    n_runs = len(df)
    #  Get generator output
    df_capacity_ratio = df.loc[:, df.columns.str.contains('Ratio of Capacity')]
    df_capacity_ratio.columns = df_capacity_ratio.columns.str.extract('(\d+)').astype(int)[0].tolist()
    df_capacity_ratio = df_capacity_ratio.transpose()
    ser_gen_capacity = pd.Series(df_gen_info['MATPOWER Capacity (MW)'].values, index=df_gen_info['MATPOWER Index'])
    df_gen_output = df_capacity_ratio.multiply(ser_gen_capacity, axis='index')
    # Combine into fuel/cooling system type
    df_gen_output = df_gen_output.merge(df_gen_info, left_index=True, right_on='MATPOWER Index')
    df_fuelcool_output_ratio = df_gen_output.groupby(['MATPOWER Fuel', '923 Cooling Type']).sum()
    # Get Ratio
    df_fuelcool_output_ratio = df_fuelcool_output_ratio.iloc[:, 0:n_runs].divide(df_fuelcool_output_ratio['MATPOWER Capacity (MW)'], axis='index') # Hardcoded
    # Combine with input factors
    df_fuelcool_output_ratio = df_fuelcool_output_ratio.transpose()
    df_fuelcool_output_ratio.columns = ['/'.join(col).strip() for col in df_fuelcool_output_ratio.columns.values]
    df_fuelcool_output_ratio = df_fuelcool_output_ratio.join(df[factor_labs+obj_labs])
    return df_fuelcool_output_ratio


def get_gen_output_ratio_with_fuelcool(df, df_gen_info):
    id_cols = ['MATPOWER Index', 'MATPOWER Fuel', '923 Cooling Type']
    #  Get fuel/cooling type
    df_capacity_ratio = df.loc[:, df.columns.str.contains('Ratio of Capacity')]  # Extract capacity ratios
    df_capacity_ratio.columns = df_capacity_ratio.columns.str.extract('(\d+)').astype(int)[0].tolist()  # Change to numeric
    df_capacity_ratio = df_capacity_ratio.transpose()
    df_capacity_ratio = df_capacity_ratio.merge(df_gen_info[id_cols], left_index=True, right_on='MATPOWER Index')  # Get cooling system/fuel types

    # Get Run information
    df_capacity_ratio = df_capacity_ratio.melt(id_vars=id_cols, value_name='Generator Output', var_name='Run ID')
    df_capacity_ratio = df_capacity_ratio.merge(df[factor_labs], left_on='Run ID', right_index=True)

    # Combine Fuel/Cooling Types
    df_capacity_ratio['Fuel/Cooling Type'] = df_capacity_ratio['MATPOWER Fuel'] + '/' + df_capacity_ratio['923 Cooling Type']
    return df_capacity_ratio


# def draw_lineplot(*args, **kwargs):
#     data = kwargs.pop('data')
#     ax = sns.lineplot(data=data, x='Generator', y='Output Capacity Ratio', style='Withdrawal Weight ($/Gallon)',
#              hue='Uniform Loading Factor', estimator=None)
#     ax.figure.canvas.draw()
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=10)
#     return ax
#
#
# def viz_6(df):
#     # Prepare Data
#     plot_df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
#     plot_df = plot_df.round({'Uniform Water Factor': 2})
#     plot_df = plot_df[plot_df['Uniform Water Factor'].isin(plot_df['Uniform Water Factor'].unique()[1::2])]
#     plot_df['Withdrawal Weight ($/Gallon)'] = pd.cut(plot_df['Withdrawal Weight ($/Gallon)'], [0.0, 0.01, 0.1001], labels=['0.0', '[0.01, 0.1]'], right=False)
#     plot_df = pd.melt(plot_df, value_vars=plot_df.columns[4: 4+49], id_vars=['Withdrawal Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor'],value_name='Output Capacity Ratio', var_name='Generator')
#     plot_df['Generator'] = plot_df['Generator'].str.extract('(\d+)')
#     # Plot
#     g = sns.FacetGrid(plot_df, col='Uniform Water Factor', col_wrap=3, aspect=2.5)
#     g.map_dataframe(draw_lineplot)
#     g.add_legend()
#     g.set_ylabels('Output Capacity Ratio')
#     g.set_xlabels('Generator')
#     plt.show()
#     return g

def draw_lineplot_wrapper(*args, **kwargs):
    data = kwargs.pop('data')
    ax = sns.lineplot(data=data, x='Uniform Loading Factor', y='Generator Output', style='MATPOWER Index', markers=True)
    #ax.figure.canvas.draw()
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=10)
    return ax


def viz_effect_of_withdrawal_weight_gen_output(df):
    df = df[df['Withdrawal Weight ($/Gallon)'] == 0.0]
    df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
    df = df[df['Uniform Water Factor'] == 1.5]

    df = df[df['Fuel/Cooling Type'] == 'coal/OC']

    # Group together generators with same behavior
    #df['Generator Output'] = df['Generator Output'].round(2)  # Assume same due to rounding
    #df = df.groupby(['Generator Output', 'Fuel/Cooling Type'] + factor_labs)['MATPOWER Index'].apply(list)
    #df = df.apply(lambda x: ', '.join([str(i) for i in x]))
    #df = df.reset_index()

    df = df[df['Fuel/Cooling Type'] == 'coal/OC']

    for i in df['Fuel/Cooling Type'].unique():
        df_plot = df[df['Fuel/Cooling Type'] == i]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=df_plot, x='Uniform Loading Factor', y='Generator Output', style='MATPOWER Index', markers=True, ax=ax, alpha=0.5)
        ax.set_title(i)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.show()

    g = sns.FacetGrid(df, col='Fuel/Cooling Type')
    g.map_dataframe(draw_lineplot_wrapper)
    g.add_legend()
    plt.show()

    return 0


def main():
    df = pd.read_csv(pathto_samples)
    df_gen_info = pd.read_csv(pathto_gen_info)
    #viz_effect_of_withdrawal_weight(df).savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Withdrawl.pdf'))
    df_capacity_ratio = get_gen_output_ratio_with_fuelcool(df, df_gen_info)

    viz_effect_of_withdrawal_weight_gen_output(df_capacity_ratio)

    #df_fuelcool_output_ratio = get_fuelcool_output_ratio(df, df_gen_info)
    #viz_6(df).fig.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Generator Output.pdf'))
    return 0


if __name__ == '__main__':
    main()
