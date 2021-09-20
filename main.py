import sys
sys.path.insert(0, 'src')
import os
import src


pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v1.0'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')
pathto_case = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.p')
pathto_geninfo = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info.csv')
pathto_geninfo_match = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'gen_info_match.csv')

def main():
    # Ordered Analysis
    src.convert_matpower(pathto_matpowercase, pathto_case)
    print('Success: convert_matpower')
    src.generator_match(pathto_geninfo, pathto_geninfo_match)
    return 0


if __name__ == '__main__':
    main()

