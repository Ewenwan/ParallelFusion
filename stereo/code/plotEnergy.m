function [plots_global, plots_thread] = plotEnergy(dataset, nt, max_time)
%% function [plots_global, plots_thread1, plots_thread2] = plotEnergy(dataset, nt)
% dataset: path to root of dataset
% nt: number of threads
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 25);
set(0, 'DefaultTextFontname', 'Times New Roman');
set(0, 'DefaultTextFontSize', 25);

method_name = {'Sequential', 'Swarn_multiway', 'Swarn', 'Victor_multiway', 'Victor', 'Hierarchy'};
legend_name_global = {'Sequential alpha-expansion', 'SF', 'SF-MF', 'SF-SS', 'Parallel alpha-expansion', 'Hierarchy'};
line_width = 2.0;
%TODO:
line_specs = {'','','','','',''};

%nm: number of methods
nm = numel(method_name);
plots_global = cell(nm);
plots_thread = cell(2);
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

%subfigure for thread
legend_name_thread = cell(nt);
for i=1:nt
    legend_name_thread{i} = sprintf('thread %d', i);
    disp(legend_name_thread{i});
end

hold off;

for i=1:2
    fig_thread = figure(i+1);
    hold on;
    mid = i*2+1;
    for j=1:nt
        filepath = sprintf('%s/temp/plot_%s_thread%d.txt', dataset, method_name{mid}, j-1);
        disp(filepath);
        thd = dlmread(filepath);
        thd_trun = thd(thd(:,1) < max_time / 2 & thd(:,1) > 0.01, :);
        plots_thread{i} = plot(thd_trun(:,1), log(thd_trun(:,2)), 'LineWidth', line_width);
    end
    legend('Thread 1', 'Thread 2', 'Thread 3', 'Thread 4');
    xlabel('Time/s');
    ylabel('Energy');
    fig_thread.Position = [500,500,640,500];
    hold off;
end
end