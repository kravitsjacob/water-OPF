
import os
from dtreeviz.trees import *
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.1'
pathto_samples = os.path.join(pathto_data, 'uniform_sa_samples', 'samples.csv')
pathto_figures = os.path.join(pathto_data, 'figures')


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
        renderPDF.drawToFile(drawing, os.path.join(pathto_figures, filenames[index]+'.pdf'))
        os.remove(filenames[index]+'.svg')
        os.remove(filenames[index])
    return 0


def main():
    df = pd.read_csv(pathto_samples)
    mods = fitSingleModels(df)
    dtreeViz(mods, df, ['Total Cost (Dollar) Tree', 'Generator Cost (Dollar) Tree', 'Water Withdrawal (Gallon) Tree', 'Water Consumption (Gallon) Tree'])
    return 0


if __name__ == '__main__':
    main()
