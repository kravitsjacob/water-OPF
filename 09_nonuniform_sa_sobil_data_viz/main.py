
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2'
pathto_operation = os.path.join(pathto_data, 'nonuniform_sa_scenarios', 'operational_scenarios.csv')
pathto_figures = os.path.join(pathto_data, 'figures')


def getHeatmap(df):
    ax = sns.heatmap(df, vmin=0.0, vmax=0.1)
    plt.show()
    return 0


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    return sns.heatmap(data.iloc[:, :-1], **kwargs)


def main():
    # Import Operational Scenarios
    df_operation = pd.read_csv(pathto_operation)

    # Import Sobol Indices
    lis_results = []
    for index, row in df_operation.iterrows():
        path = os.path.join(pathto_data, 'nonuniform_sa_sobol', row['Operational Scenario']+' sobol.csv')
        df_results = pd.read_csv(path, index_col=0)
        df_results['Scenario'] = row['Operational Scenario']
        lis_results.append(df_results)
    df_results = pd.concat(lis_results)

    # Filter
    df_results[df_results._get_numeric_data() < -1] = np.nan  # Invalid Index

    # Plotting
    df_results.columns = ['$C_{water,coal,once-through}$', '$C_{water,coal,recirculating}$',
                          '$C_{water,coal,induced-recirculating}$', '$C_{water,natural-gas,induced-recirculating}$',
                          '$C_{water,nuclear,recirculating}$', 'Scenario']

    g = sns.FacetGrid(df_results, col='Scenario', height=6.5, aspect=0.35, sharey=True)
    cbar_ax = g.fig.add_axes([.87, .15, .03, .7])
    g.map_dataframe(draw_heatmap, cmap='viridis', cbar_ax=cbar_ax, cbar_kws={'label': 'First Order Sobol Index Value'})
    g.fig.subplots_adjust(top=0.6, right=0.85)
    g.set_titles(rotation=90)
    axes = g.axes.flatten()
    axes[0].set_title('Normal loading with \n traditional OPF (BAU policy)')
    axes[1].set_title('Heatwave with traditional \n OPF (BAU policy)')
    axes[2].set_title('Heatwave with aggressive  \n withdrawal policy')
    axes[3].set_title('Heatwave with aggressive \n consumption policy')
    plt.savefig(os.path.join(pathto_figures, 'First Order Heatmap.pdf'))
    plt.show()
    return 0


if __name__ == '__main__':
    main()
