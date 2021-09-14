
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2'
pathto_samples = os.path.join(pathto_data, 'uniform_sa_samples', 'samples.csv')
pathto_figures = os.path.join(pathto_data, 'figures')

factor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor']
obj_labs = ['Total Cost ($)', 'Generator Cost ($)',	'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)']


def viz_5(df):
    # Prepare Data
    plot_df = df[factor_labs + obj_labs]
    plot_df = plot_df[plot_df['Consumption Weight ($/Gallon)'] == 0.0]
    plot_df = plot_df.round({'Uniform Water Factor': 2})
    plot_df = plot_df[plot_df['Uniform Water Factor'].isin(plot_df['Uniform Water Factor'].unique()[1::2])]  # Select every other loading factor value
    plot_df['Withdrawal Weight ($/Gallon)'] = pd.cut(plot_df['Withdrawal Weight ($/Gallon)'], [0.0, 0.01, 0.1], labels=['0.0', '[0.01, 0.1]'], right=False)
    # Plot
    plot_df = pd.melt(plot_df, value_vars='Water Withdrawal (Gallon)', id_vars=factor_labs)
    g = sns.FacetGrid(plot_df, col='Uniform Water Factor')
    g.map(sns.scatterplot, 'Uniform Loading Factor', 'value', 'Withdrawal Weight ($/Gallon)')
    g.add_legend(title='Binned Withdrawal Weight ($/Gallon)')
    g.set_ylabels('Water Withdrawal (Gallon)')
    plt.show()
    return g


def draw_lineplot(*args, **kwargs):
    data = kwargs.pop('data')
    ax = sns.lineplot(data=data, x='Generator', y='Output Capacity Ratio', style='Withdrawal Weight ($/Gallon)',
             hue='Uniform Loading Factor', estimator=None)
    ax.figure.canvas.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=10)
    return ax


def viz_6(df):
    # Prepare Data
    plot_df = df[df['Consumption Weight ($/Gallon)'] == 0.0]
    plot_df = plot_df.round({'Uniform Water Factor': 2})
    plot_df = plot_df[plot_df['Uniform Water Factor'].isin(plot_df['Uniform Water Factor'].unique()[1::2])]
    plot_df['Withdrawal Weight ($/Gallon)'] = pd.cut(plot_df['Withdrawal Weight ($/Gallon)'], [0.0, 0.01, 0.1001], labels=['0.0', '[0.01, 0.1]'], right=False)
    plot_df = pd.melt(plot_df, value_vars=plot_df.columns[4: 4+49], id_vars=['Withdrawal Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor'],value_name='Output Capacity Ratio', var_name='Generator')
    plot_df['Generator'] = plot_df['Generator'].str.extract('(\d+)')
    # Plot
    g = sns.FacetGrid(plot_df, col='Uniform Water Factor', col_wrap=3, aspect=2.5)
    g.map_dataframe(draw_lineplot)
    g.add_legend()
    g.set_ylabels('Output Capacity Ratio')
    g.set_xlabels('Generator')
    plt.show()
    return g


def main():
    df = pd.read_csv(pathto_samples)
    viz_5(df).fig.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Withdrawl.pdf'))
    viz_6(df).fig.savefig(os.path.join(pathto_figures, 'Effect of Withdrawal Weight on Generator Output.pdf'))
    return 0


if __name__ == '__main__':
    main()
