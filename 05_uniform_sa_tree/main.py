
import os
import pandas as pd
from sklearn import tree

from dtreeviz.trees import *

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# import graphviz



pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\Water_OPF_GS_V5_io'
pathto_results = os.path.join(pathto_data, 'output', 'results.csv')

factor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor']
obj_labs = ['Total Cost ($)', 'Generator Cost ($)',	'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)']


def fitSingleModels(df):
    mods = {}
    for i in obj_labs:
        clf = tree.DecisionTreeRegressor(random_state=1008, max_depth=5, max_leaf_nodes=12)
        mods[i] = clf.fit(X=df[factor_labs], y=df[i])
    return mods


def dtreeViz(mods, df, filenames):
    for index, key in enumerate(mods):
        viz = dtreeviz(mods[key],
                       df[factor_labs],
                       df[key],
                       target_name=key,
                       feature_names=factor_labs,
                       orientation='LR')
        viz.save(filenames[index]+'.svg')
        drawing = svg2rlg(filenames[index]+'.svg')
        renderPDF.drawToFile(drawing, filenames[index]+'.pdf')
        os.remove(filenames[index]+'.svg')
        os.remove(filenames[index])
    return 0


def dtreeVizHeatmap(df, filenames):
    for i, j in enumerate(obj_labs):
        clf = tree.DecisionTreeRegressor(random_state=1008, max_depth=5, max_leaf_nodes=12)
        clf = clf.fit(X=df[['Uniform Water Factor', 'Uniform Loading Factor']], y=df[j])
        # Heatmap
        rtreeviz_bivar_heatmap(clf,
                               df[['Uniform Water Factor', 'Uniform Loading Factor']],
                               df[j],
                               feature_names=['Uniform Water Factor', 'Uniform Loading Factor'],
                               show={}
                               )
        plt.savefig(filenames[i] + ' 2D Heatmap.pdf')
        plt.show()
        # 3D
        rtreeviz_bivar_3D(clf,
                          df[['Uniform Water Factor', 'Uniform Loading Factor']],
                          df[j],
                          target_name=j,
                          feature_names=['Uniform Water Factor', 'Uniform Loading Factor'],
                          elev=20,
                          azim=10,
                          dist=8.2,
                          show={}
                          )
        plt.savefig(filenames[i] + ' 3D Heatmap.pdf')
        plt.show()
    return 0


def vizModels(mods):
    graphs = {}
    for index, key in enumerate(mods):
        dot_data = tree.export_graphviz(mods[key], feature_names=factor_labs, class_names=key, filled=True)
        graphs[key] = graphviz.Source(dot_data)
    return graphs


def renderGraphs(graphs, filenames):
    for index, key in enumerate(graphs):
        graphs[key].render(filenames[index])
        os.remove(filenames[index])
    return 0

def main():
    df = pd.read_csv(pathto_results)
    # Single-output Decision Trees
    mods = fitSingleModels(df)

    dtreeVizHeatmap(df, ['Total Cost (Dollar)', 'Generator Cost (Dollar)', 'Water Withdrawal (Gallon)',
                        'Water Consumption (Gallon)'])
    dtreeViz(mods, df, ['Total Cost (Dollar) Tree', 'Generator Cost (Dollar) Tree', 'Water Withdrawal (Gallon) Tree', 'Water Consumption (Gallon) Tree'])


    # graphs = vizModels(mods)
    # renderGraphs(graphs, ['Total Cost (Dollar) Tree', 'Generator Cost (Dollar) Tree', 'Water Withdrawal (Gallon) Tree', 'Water Consumption (Gallon) Tree'])
    # Multi-output Decision Tree
    #multimod = fitMultiModel(df)
    return 0


if __name__ == '__main__':
    main()
