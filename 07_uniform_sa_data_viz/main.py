
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import hiplot as hip


pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\Water_OPF_GS_V5_io'
pathto_results = os.path.join(pathto_data, 'output', 'results.csv')

factor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor']
obj_labs = ['Total Cost ($)', 'Generator Cost ($)',	'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)']


def viz_1(df):
    # Prepare Data
    plot_df = pd.melt(df, value_vars=obj_labs, id_vars=factor_labs, value_name='Objective Value', var_name='Objective')
    plot_df = pd.melt(plot_df, value_vars=factor_labs, id_vars=['Objective Value', 'Objective'], var_name='Input Factor',
                value_name='Input Factor Value')
    # Plot
    g = sns.FacetGrid(plot_df, row='Objective', col='Input Factor', sharey='row', sharex='col')
    g.map(sns.scatterplot, 'Input Factor Value', 'Objective Value')
    [ax.set_title('') for ax in g.axes.flat]
    g.axes[0, 0].set_ylabel('Foo')
    [ax.set_ylabel(obj_labs[i]) for i, ax in enumerate(g.axes[:, 0])]
    [ax.set_xlabel(factor_labs[i]) for i, ax in enumerate(g.axes[3, :])]
    return g


def viz_2(df):
    for i in obj_labs:
        # Plot
        plot_df = df[factor_labs + [i]]
        g = sns.PairGrid(plot_df, hue=i, palette='viridis', corner=True)
        g.map_lower(sns.scatterplot)
        [g.axes[i, i].set_visible(False) for i in [0, 1, 2, 3]]
        # Colorbar
        cbar_ax = g.fig.add_axes([.60, .3, .05, .4])
        norm = plt.Normalize(plot_df[i].min(), plot_df[i].max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        g.fig.figure.colorbar(sm, cax=cbar_ax)
        cbar_ax.set_ylabel(i)
        plt.show()
    return g


def viz_3(df):
    for i in obj_labs:
        a = 1
        plot_df = df[factor_labs + [i]]
        plot_df = pd.melt(plot_df, value_vars=i, id_vars=factor_labs, value_name='Objective Value', var_name='Objective')
        # Plot
        g = sns.FacetGrid(plot_df, row='Withdrawal Weight ($/Gallon)', col='Consumption Weight ($/Gallon)')
        g.map(sns.scatterplot, 'Uniform Water Factor', 'Uniform Loading Factor', 'Objective Value')
        plt.show()
    return g


def viz_4(df):
    plot_df = df[factor_labs + obj_labs]
    # Create Plot
    exp = hip.Experiment.from_dataframe(plot_df)
    exp.parameters_definition['Total Cost ($)'].colormap = 'interpolateViridis'
    exp.display_data(hip.Displays.PARALLEL_PLOT).update({'hide': ['uid']})
    exp.display_data(hip.Displays.TABLE).update({'hide': ['uid', 'from_uid']})
    exp.to_html('parallel.html')
    return 0


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
    plt.savefig('Effect of Withdrawal Weight on Withdrawl.pdf')
    plt.show()
    # All Vars
    # plot_df = pd.melt(plot_df, value_vars=obj_labs, id_vars=factor_labs, value_name='Objective Value', var_name='Objective')
    # g = sns.FacetGrid(plot_df, row='Objective', col='Uniform Water Factor')
    # g.map(sns.scatterplot, 'Uniform Loading Factor', 'Objective Value', 'Withdrawal Weight ($/Gallon)')
    plt.show()
    return 0


def draw_lineplot(*args, **kwargs):
    data = kwargs.pop('data')
    ax = sns.lineplot(data=data, x='Generator', y='Output Capacity Ratio', style='Withdrawal Weight ($/Gallon)',
             hue='Uniform Loading Factor', estimator=None)
    ax.figure.canvas.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=10)
    return ax

# fig, ax = plt.subplots(figsize=(20,4))
# sns.scatterplot(data=data, x='Generator', y='Output Capacity Ratio', hue='Withdrawal Weight ($/Gallon)', estimator=None, ax=ax)
# plt.show()


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
    plt.savefig('Effect of Withdrawal Weight on Generator Output.pdf')
    plt.show()
    return 0


def main():
    df = pd.read_csv(pathto_results)
    #viz_1(df)
    #viz_2(df)
    #viz_3(df)
    #viz_4(df)
    #viz_5(df)
    viz_6(df)

    return 0


if __name__ == '__main__':
    main()
