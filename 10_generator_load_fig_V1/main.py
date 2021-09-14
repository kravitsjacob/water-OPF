
import os
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\load exogenous parameter testing V1 io'
pathto_load = os.path.join(pathto_data, '20180101-20200101 MISO Forecasted Cleared & Actual Load.csv')
pathto_figures = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2\\figures'

def importData():
    df = pd.read_csv(pathto_load)
    df = df.rename({'ActualLoad': 'Actual Load (MW)'}, axis='columns')
    return df


def plotter(df):
    df['Uniform Loading Coefficient'] = df['Actual Load (MW)'] / df['Actual Load (MW)'].median()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    sns.histplot(data=df, x='Actual Load (MW)', ax=ax1)
    sns.histplot(data=df, x='Uniform Loading Coefficient', ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(pathto_figures, 'Load Distribution.pdf'))
    plt.show()
    return 0


def main():
    df = importData()
    plotter(df)
    return 0


if __name__ == '__main__':
    main()
