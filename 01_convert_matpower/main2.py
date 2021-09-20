import pandapower.converter
import os

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2\synthetic_grid'
pathto_matpowercase = os.path.join(pathto_data, 'case.mat')
pathto_case = os.path.join(pathto_data, 'case.p')

net = pandapower.converter.from_mpc(pathto_matpowercase)
pandapower.to_pickle(net, pathto_case)
