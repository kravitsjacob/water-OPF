clc


pathto_case = fullfile('G:\My Drive\Documents (Stored)\data_sets\Illinois Synthetic Grid\ACTIVSg200', 'case_ACTIVSg200.m');
pathto_case_info = fullfile('G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.1\synthetic_grid', 'gen_info.csv');
pathto_case_export = fullfile('G:\My Drive\Documents (Stored)\data_sets\water-OPF-v0.1\synthetic_grid', 'case.mat');


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
    T = table(mpc.gen(:,1), 'VariableNames', {'Synthetic Generator Index'});
    T.('Synthetic Generator Name') = mpc.bus_name(T.('Synthetic Generator Index'));
    T.('Synthetic Type') = mpc.gentype;
    T.('Synthetic Fuel') = mpc.genfuel;
    T.('Synthetic Maximum real power output (MW)') = mpc.gen(:,9); 
end
