pathto_case='../water-OPF_io/external_data/Illinois Synthetic Grid/ACTIVSg200/case_ACTIVSg200.m'
pathto_case_info='../water-OPF_io/temp/synthetic_grid/gen_info.csv',
pathto_case_export='../water-OPF_io/temp/synthetic_grid/case.mat'

main(pathto_case, pathto_case_info, pathto_case_export)

function main(pathto_case, pathto_case_info, pathto_case_export)
    % Load the Case
    pathto_case
    mpc = loadcase(pathto_case);
    % Save Case
    savecase(pathto_case_export, mpc)
    % Extract Test Case Information
    T = table(mpc.gen(:,1), 'VariableNames', {'MATPOWER Index'});
    T.('MATPOWER Generator Name') = mpc.bus_name(T.('MATPOWER Index'));
    T.('MATPOWER Type') = mpc.gentype;
    T.('MATPOWER Fuel') = mpc.genfuel;
    T.('MATPOWER Capacity (MW)') = mpc.gen(:,9);
    % Write Table
    writetable(T, pathto_case_info)
    % Save Case
    savecase(pathto_case_export, mpc)
end


