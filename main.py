import sys
sys.path.insert(0, 'src')
import os
import src


pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v1.0'
pathto_matpowercase = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.mat')
pathto_case = os.path.join(pathto_data, 'temp', 'synthetic_grid', 'case.p')

def main():
    # Ordered Flow
    src.convert_matpower(pathto_matpowercase, pathto_case)
    print('Success: convert_matpower')

    return 0


if __name__ == '__main__':
    main()

