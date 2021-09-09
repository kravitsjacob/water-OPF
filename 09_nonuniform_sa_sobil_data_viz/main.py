
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\Water_OPF_Non-Uniform_Sobil_V3_io'
pathto_operation = os.path.join(pathto_data, 'input', 'operational_scenarios.csv')


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
    # Import Sobil Indices
    lis_results = []
    for index, row in df_operation.iterrows():
        path = os.path.join(pathto_data, 'output', 'sobil', row['Operational Scenario'] + '_results.csv')
        df_results = pd.read_csv(path, index_col=0)
        df_results['Scenario'] = row['Operational Scenario']
        lis_results.append(df_results)
    df_results = pd.concat(lis_results)
    # Plotting
    plt.figure(figsize=(15, 15))
    g = sns.FacetGrid(df_results, col='Scenario', height=5, aspect=0.4, sharey=True)
    cbar_ax = g.fig.add_axes([.91, .15, .03, .7])
    g.map_dataframe(draw_heatmap, cmap='viridis', cbar_ax=cbar_ax, cbar_kws={'label': 'First Order Sobil Index Value'})
    g.fig.subplots_adjust(top=.55, right=0.90)
    g.set_titles(rotation=90)
    axes = g.axes.flatten()
    axes[0].set_title('Normal Loading \n No Water Weight')
    axes[1].set_title('Normal Loading \n High Withdrawal Weight')
    axes[2].set_title('Normal Loading \n High Consumption Weight')
    axes[3].set_title('Extreme Heatwave \n No Water Weight')
    axes[4].set_title('Extreme Heatwave \n High Withdrawal Weight')
    axes[5].set_title('Extreme Heatwave \n High Consumption Weight')
    plt.savefig('First Order Heatmap.pdf')
    plt.show()
    return 0


if __name__ == '__main__':
    main()
