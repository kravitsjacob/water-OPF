clc


pathto_case = fullfile('G:\My Drive\Documents (Stored)\data_sets\Illinois Synthetic Grid\ACTIVSg200', 'case_ACTIVSg200.m');
pathto_case_info = fullfile('G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2\synthetic_grid', 'gen_info.csv');
pathto_case_export = fullfile('G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.2\synthetic_grid', 'case.mat');


% Main
% Load the Case
mpc = loadcase(pathto_case);
% Save Case
savecase(pathto_case_export, mpc)
% Extract Test Case Information
T = getInfo(mpc);
% Write Table
writetable(T, pathto_case_info)
% Save Case
savecase(pathto_case_export, mpc)


function T = getInfo(mpc)
    T = table(mpc.gen(:,1), 'VariableNames', {'MATPOWER Index'});
    T.('MATPOWER Generator Name') = mpc.bus_name(T.('MATPOWER Index'));
    T.('MATPOWER Type') = mpc.gentype;
    T.('MATPOWER Fuel') = mpc.genfuel;
    T.('MATPOWER Capacity (MW)') = mpc.gen(:,9); 
end
