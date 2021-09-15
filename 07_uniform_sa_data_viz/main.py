
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


def get_coolfuel_output_ratio(df, df_gen_info):
    #  Get generator output
    df_capacity_ratio = df.iloc[:, 4: 4 + 49]
    df_capacity_ratio.columns = df_capacity_ratio.columns.str.extract('(\d+)').astype(int)[0].tolist()
    df_capacity_ratio = df_capacity_ratio.transpose()
    ser_gen_capacity = pd.Series(df_gen_info['MATPOWER Capacity (MW)'].values, index=df_gen_info['MATPOWER Index'])
    df_capacity_ratio.multiply(ser_gen_capacity, axis='index')

    # Combine into fuel/cooling system type

    # Pivot table
    return 0


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


def main():
    df = pd.read_csv(pathto_samples)
    df_gen_info = pd.read_csv(pathto_gen_info)
    #viz_effect_of_withdrawal_weight(df).savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Withdrawl.pdf'))
    get_coolfuel_output_ratio(df, df_gen_info)
    #viz_6(df).fig.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Generator Output.pdf'))
    return 0


if __name__ == '__main__':
    main()
