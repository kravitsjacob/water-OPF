import pandapower.converter


def convert_matpower(pathto_matpowercase, pathto_case):
    net = pandapower.converter.from_mpc(pathto_matpowercase)
    pandapower.to_pickle(net, pathto_case)
    return 0