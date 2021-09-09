import pandapower.converter
import os

pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\Convert Matpower V3_io'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'case.mat')
pathto_case = os.path.join(pathto_data, 'output', 'case.p')

net = pandapower.converter.from_mpc(pathto_matpowercase)
pandapower.to_pickle(net, pathto_case)
