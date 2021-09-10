import osimport copyimport sysimport itertoolsimport numpy as npimport pandas as pdimport pandapower as ppimport pandapower.converterimport dask.dataframe as ddimport multiprocessingfrom dask.diagnostics import ProgressBarif len(sys.argv) > 1:    pathto_data = sys.argv[1]    n_tasks = int(sys.argv[2])else:    pathto_data = 'G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.1'    n_tasks = os.cpu_count()pathto_geninfo = os.path.join(pathto_data, 'synthetic_grid', 'gen_info_match_water.csv')pathto_case = os.path.join(pathto_data, 'synthetic_grid', 'case.p')pathto_output = os.path.join(pathto_data, 'uniform_sa_samples', 'samples.csv')def initialize():    # Import Data    df_geninfo = pd.read_csv(pathto_geninfo)    net = pandapower.from_pickle(pathto_case)    # Match Pandapower indexes to Matpower    df_geninfo = matchIndex(df_geninfo, net)    # Initialize Labels    return df_geninfo, netdef uniformFactorMultiply(c_water, c_load, beta_with, beta_con, beta_load, exogenous_labs):    vals = np.concatenate((c_water * beta_with, c_water * beta_con, c_load * beta_load))    idxs = exogenous_labs[2:]    return pd.Series(vals, index=idxs)def getSearch(df_gridspecs, df_geninfo, net, exogenous_labs):    # Set Search Values    w_with_vals = np.linspace(df_gridspecs['Min']['Withdrawal Weight ($/Gallon)'],                              df_gridspecs['Max']['Withdrawal Weight ($/Gallon)'],                              df_gridspecs['Number of Steps']['Withdrawal Weight ($/Gallon)'])    w_con_vals = np.linspace(df_gridspecs['Min']['Consumption Weight ($/Gallon)'],                             df_gridspecs['Max']['Consumption Weight ($/Gallon)'],                             df_gridspecs['Number of Steps']['Withdrawal Weight ($/Gallon)'])    c_load_vals = np.linspace(df_gridspecs['Min']['Uniform Loading Factor'],                              df_gridspecs['Max']['Uniform Loading Factor'],                              df_gridspecs['Number of Steps']['Uniform Loading Factor'])    c_water_vals = np.linspace(df_gridspecs['Min']['Uniform Water Factor'],                               df_gridspecs['Max']['Uniform Water Factor'],                               df_gridspecs['Number of Steps']['Uniform Water Factor'])    # Create Grid    df_search = pd.DataFrame(list(itertools.product(w_with_vals, w_con_vals, c_load_vals, c_water_vals)), columns=df_gridspecs.index)    # Multiply exogenous parameters    beta_with = df_geninfo['Median Withdrawal Rate (Gallon/kWh)'].values    beta_con = df_geninfo['Median Consumption Rate (Gallon/kWh)'].values    beta_load = net.load['p_mw'].values    df_exogenous = df_search.apply(lambda row: uniformFactorMultiply(row['Uniform Water Factor'], row['Uniform Loading Factor'], beta_with, beta_con, beta_load, exogenous_labs), axis=1)    # Combine    df_search = pd.concat([df_search, df_exogenous], axis=1)    return df_searchdef matchIndex(df_geninfo, net):    # Initialize    df_gen = pd.DataFrame()    df_sgen = pd.DataFrame()    df_ext_grid = pd.DataFrame()    # Generators    df_gen['MATPOWER Index'] = net.gen['bus'] + 1    df_gen['PANDAPOWER Index'] = net.gen['bus'].index.to_list()    df_gen['PANDAPOWER Bus Type'] = 'gen'    # Static Generators    df_sgen['MATPOWER Index'] = net.sgen['bus'] + 1    df_sgen['PANDAPOWER Index'] = net.sgen['bus'].index.to_list()    df_sgen['PANDAPOWER Bus Type'] = 'sgen'    # External Grid    df_ext_grid['MATPOWER Index'] = net.ext_grid['bus'] + 1    df_ext_grid['PANDAPOWER Index'] = net.ext_grid['bus'].index.to_list()    df_ext_grid['PANDAPOWER Bus Type'] = 'ext_grid'    # Combine    df_match = pd.concat([df_gen, df_sgen, df_ext_grid])    df_match.reset_index()    # Join    df_geninfo = df_geninfo.merge(df_match, on='MATPOWER Index')    return df_geninfodef waterOPF(ser_exogenous, t, results_labs, net, df_geninfo):    # Initialize    df_geninfo = df_geninfo.copy()    net = copy.deepcopy(net)  # Copy network so not changed later    obj_colnames = ['cp0_eur', 'cp1_eur_per_mw', 'cp2_eur_per_mw2']    idx_colnames = ['element', 'et']    # Create DataFrame of loads    df_load = ser_exogenous[ser_exogenous.index.str.contains('Load')].to_frame('Load (MW)')    df_load['bus'] = df_load.index.str.extract('(\d+)').astype(int).values    df_load.reset_index(inplace=True, drop=True)    # Create DataFrame of cost values    df_cost = net.poly_cost[idx_colnames + obj_colnames]    # Create DataFrame of withdrawal values    df_withdrawal = ser_exogenous[ser_exogenous.index.str.contains('Withdrawal Rate')].to_frame('Withdrawal Rate (Gallon/kWh)')    df_withdrawal['MATPOWER Index'] = df_withdrawal.index.str.extract('(\d+)').astype(int).values    df_withdrawal.reset_index(inplace=True, drop=True)    # Create DataFrame of consumption values    df_consumption = ser_exogenous[ser_exogenous.index.str.contains('Consumption Rate')].to_frame('Consumption Rate (Gallon/kWh)')    df_consumption['MATPOWER Index'] = df_consumption.index.str.extract('(\d+)').astype(int).values    df_consumption.reset_index(inplace=True, drop=True)    # Write objectives information to generator information    df_geninfo[['Cost Term ($)', 'Cost Term ($/MW)', 'Cost Term ($/MW^2)']] = df_geninfo.merge(df_cost, left_on=['PANDAPOWER Index', 'PANDAPOWER Bus Type'], right_on=idx_colnames)[obj_colnames]    df_geninfo['Withdrawal Rate (Gallon/kWh)'] = df_geninfo.merge(df_withdrawal, on='MATPOWER Index')['Withdrawal Rate (Gallon/kWh)']    df_geninfo['Consumption Rate (Gallon/kWh)'] = df_geninfo.merge(df_consumption, on='MATPOWER Index')['Consumption Rate (Gallon/kWh)']    # Convert Units    df_geninfo['Withdrawal Rate (Gallon/MW)'] = df_geninfo['Withdrawal Rate (Gallon/kWh)'] * t  # minutes * hr/minutes * kw/MW    df_geninfo['Consumption Rate (Gallon/MW)'] = df_geninfo['Consumption Rate (Gallon/kWh)'] * t  # minutes * hr/minutes * kw/MW    # Combine and weight objectives    df_geninfo['Weighted Linear Term'] = df_geninfo['Cost Term ($/MW)'] + df_geninfo['Withdrawal Rate (Gallon/MW)'] * ser_exogenous['Withdrawal Weight ($/Gallon)'] + df_geninfo['Consumption Rate (Gallon/MW)'] * ser_exogenous['Consumption Weight ($/Gallon)']    # Assign exogenous parameters to case    net.poly_cost[obj_colnames] = net.poly_cost.merge(df_geninfo, left_on=idx_colnames, right_on=['PANDAPOWER Index', 'PANDAPOWER Bus Type'])[['Cost Term ($)', 'Weighted Linear Term', 'Cost Term ($/MW^2)']]    net.load['p_mw'] = net.load.merge(df_load)['Load (MW)']    # Run DC OPF    try:        pp.rundcopp(net)        state = 'converge'    except:        state = 'not converge'    # Output Depended on State    if state is 'converge':        # Extract internal decisions (power output)        for type in ['gen', 'sgen', 'ext_grid']:            idxs = df_geninfo.index[df_geninfo['PANDAPOWER Bus Type'] == type]            df_geninfo.loc[idxs, 'Power Output (MW)'] = df_geninfo.iloc[idxs, :].merge(net['res_'+type], left_on='PANDAPOWER Index', right_index=True)['p_mw']        # Compute Capacity Ratios        df_geninfo['Ratio of Capacity'] = df_geninfo['Power Output (MW)'] / df_geninfo['MATPOWER Capacity (MW)']        # Compute Objectives        F_gen = (df_geninfo['Cost Term ($)'] + df_geninfo['Power Output (MW)'] * df_geninfo['Cost Term ($/MW)'] + df_geninfo['Power Output (MW)']**2 * df_geninfo['Cost Term ($/MW^2)']).sum()        F_with = (df_geninfo['Power Output (MW)'] * df_geninfo['Withdrawal Rate (Gallon/MW)']).sum()        F_con = (df_geninfo['Power Output (MW)'] * df_geninfo['Consumption Rate (Gallon/MW)']).sum()        F_cos = F_gen + ser_exogenous['Withdrawal Weight ($/Gallon)']*F_with + ser_exogenous['Consumption Weight ($/Gallon)']*F_con        internal_decs = df_geninfo['Ratio of Capacity'].to_list()    elif state is 'not converge':        F_cos = F_with = F_con = F_gen = np.nan        internal_decs = [np.nan]*len(df_geninfo)    return pd.Series([F_cos, F_gen, F_with, F_con] + internal_decs, index=results_labs)def main():    # Initialize    df_geninfo, net = initialize()    exogenous_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)'] + \                     ('MATPOWER Generator ' + df_geninfo['MATPOWER Index'].astype(str) + ' Withdrawal Rate (Gallon/kWh)').tolist() + \                     ('MATPOWER Generator ' + df_geninfo['MATPOWER Index'].astype(str) + ' Consumption Rate (Gallon/kWh)').tolist() + \                     ('PANDAPOWER Bus ' + net.load['bus'].astype(str) + ' Load (MW)').tolist()    inputfactor_labs = ['Withdrawal Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Loading Factor', 'Uniform Water Factor']    results_labs = ['Total Cost ($)', 'Generator Cost ($)', 'Water Withdrawal (Gallon)', 'Water Consumption (Gallon)'] + \                   ('MATPOWER Generator ' + df_geninfo['MATPOWER Index'].astype(str) + ' Ratio of Capacity').to_list()    df_gridspecs = pd.DataFrame(data=[[0.0, 0.1, 10], [0.0, 1.0, 10], [1.0, 1.5, 10], [0.5, 1.5, 10]],                                index=inputfactor_labs, columns=['Min', 'Max', 'Number of Steps'])    t = 5 * 1 / 60 * 1000  # minutes * hr/minutes * kw/MW    print('Success: Initialized')    # Get Grid    df_search = getSearch(df_gridspecs, df_geninfo, net, exogenous_labs)    print('Number of Searches: ', len(df_search))    print('Success: Grid Created')    # Run Search    ddf_search = dd.from_pandas(df_search, npartitions=n_tasks)    df_results = ddf_search.apply(lambda row: waterOPF(row[exogenous_labs], t, results_labs, net, df_geninfo), axis=1, meta={key:'float64' for key in results_labs}).compute(scheduler='processes')    df_master = pd.concat([df_results, df_search], axis=1)    df_master.drop_duplicates()  # Sometimes the parallel jobs replicate rows    print('Success: Grid Searched')    df_master.to_csv(pathto_output, index=False)    return 0if __name__ == '__main__':    multiprocessing.freeze_support()    ProgressBar().register()    main()