import sys
sys.path.insert(0, 'src')
import os
import src

import pandapower.converter
import pandas as pd


pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v1.0'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')
pathto_case = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.p')
pathto_geninfo = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info.csv')
pathto_geninfo_match = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info_match.csv')


def main():

    # Converting matpower
    # net = pandapower.converter.from_mpc(pathto_matpowercase)
    # print('Success: convert_matpower')
    # pandapower.to_pickle(net, pathto_case)  # Save checkpoint
    net = pandapower.from_pickle(pathto_case)  # Import checkpoint

    # Manual generator matching
    # df_gen_info = pd.read_csv(pathto_geninfo)
    # df_gen_info_match = src.generator_match(df_gen_info)
    # print('Success: generator_match')
    # df_gen_info_match.to_csv(pathto_geninfo_match, index=False)  # Save checkpoint
    df_gen_info_match = pd.read_csv(pathto_geninfo_match)  # Import checkpoint

    a = 1
    return 0


if __name__ == '__main__':
    main()

