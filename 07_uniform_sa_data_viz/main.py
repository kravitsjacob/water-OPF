
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


def viz_effect_of_withdrawal_weight_gen_output(df):
    # Formatting
    df['MATPOWER Index'] = df['MATPOWER Index'].astype('str')

    # Subsetting Data
    df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
    withdrawal_criteria = [0.0, 0.1, 0.1]
    uniform_water_criteria = [0.5, 0.5, 1.5]
    case_a = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[0]) & (df['Uniform Water Factor'] == uniform_water_criteria[0])
    case_b = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[1]) & (df['Uniform Water Factor'] == uniform_water_criteria[1])
    case_c = (df['Withdrawal Weight ($/Gallon)'] == withdrawal_criteria[2]) & (df['Uniform Water Factor'] == uniform_water_criteria[2])
    df = df[case_a | case_b | case_c]


    # Creating Plot
    df['Withdrawal Weight ($/Gallon)'] = df['Withdrawal Weight ($/Gallon)'].astype('category')
    df = df.sort_values(['Fuel/Cooling Type', 'Uniform Loading Factor'], ascending=False)
    df['ID'] = '$w_{with}=$' + df['Withdrawal Weight ($/Gallon)'].astype(str) + ', $c_{water}=$' + df['Uniform Water Factor'].astype(str)
    g = sns.FacetGrid(df, row='ID', col='Fuel/Cooling Type', aspect=1, sharex='col')
    g.map_dataframe(sns.scatterplot, y='Generator Output', x='MATPOWER Index', hue='Uniform Loading Factor',
                    size='Uniform Loading Factor', sizes=(20, 200), style='Withdrawal Weight ($/Gallon)', linewidth=0)
    g.set_xticklabels(rotation=90)
    g.add_legend()
    g.fig.subplots_adjust(wspace=.05, hspace=.05)
    for row in range(g.axes.shape[0]):
        for col in range(g.axes.shape[1]):
            if row == 0:
                g.axes[row, col].set_title(df['Fuel/Cooling Type'].unique()[col])
            else:
                g.axes[row, col].set_title('')

            if col == 0:
                g.axes[row, col].set_ylabel('$c_{water}=$'+str(uniform_water_criteria[row]))
    plt.show()
    return g


def main():
    df = pd.read_csv(pathto_samples)
    df_gen_info = pd.read_csv(pathto_gen_info)
    viz_effect_of_withdrawal_weight(df).savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Withdrawl.pdf'))
    df_capacity_ratio = get_gen_output_ratio_with_fuelcool(df, df_gen_info)
    viz_effect_of_withdrawal_weight_gen_output(df_capacity_ratio).fig.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Generator Output.pdf'))
    return 0


if __name__ == '__main__':
    main()
