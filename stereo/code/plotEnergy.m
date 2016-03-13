function [plots_global, plots_thread1, plots_thread2] = plotEnergy(dataset, nt, max_time)
%% function [plots_global, plots_thread1, plots_thread2] = plotEnergy(dataset, nt)
% dataset: path to root of dataset
% nt: number of threads
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontname', 'Times New Roman');
set(0, 'DefaultTextFontSize', 11);

method_name = {'Sequential', 'Swarn_multiway', 'Swarn', 'Victor', 'Victor_multiway', 'Hierarchy'};
legend_name_global = {'Sequential alpha-expansion', 'SF', 'SF-MF', 'Parallel alpha-expansion', 'SF-SS', 'Hierarchy'};
line_width = 1.2;
%TODO:
line_specs = [];

%nm: number of methods
nm = numel(method_name);
plots_global = cell(nm);
%draw global energy

fig_glb = figure(1);
hold on;
for i=1:nm
    filepath = sprintf('%s/temp/plot_%s_global.txt', dataset, method_name{i});
    disp(filepath);
    glb = dlmread(filepath);
    glb_trun = glb(glb(:,1) < max_time & glb(:,1) > 0.01, :);
    plots_global{i} = plot(glb_trun(:,1), log(glb_trun(:,2)), 'LineWidth', line_width);
end
legend(legend_name_global);
xlabel('Time/s');
ylabel('Energy');
fig_glb.Position = [500,500,1280,720];
end