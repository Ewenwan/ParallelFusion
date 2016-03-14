function plotEnergy(dataset, nt, max_time)
%% function [plots_global, plots_thread1, plots_thread2] = plotEnergy(dataset, nt)
% dataset: path to root of dataset
% nt: number of threads
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize',34);
set(0, 'DefaultTextFontname', 'Times New Roman');
set(0, 'DefaultTextFontSize', 34);

method_name = {'Sequential', 'Victor', 'Hierarchy', 'Swarn', 'Victor_multiway', 'Swarn_multiway'};
legend_name_global = {'AE', 'PAE', 'HF', 'SF-MF(ours)', 'SF-SS(ours)', 'SF(ours)'};
line_width = 4.0;
%TODO:
line_specs = {'--','--','--','','',''};

%nm: number of methods
nm = numel(method_name);
plots_global = cell(nm);
plots_thread = cell(2);
%draw global energy

fig_glb = figure(1);
hold on;
for i=1:nm
    filepath = sprintf('%s/plot_%s_global.txt', dataset, method_name{i});
    disp(filepath);
    glb = dlmread(filepath);
    glb_trun = glb(glb(:,1) < max_time & glb(:,1) > 0.01, :);
    plot(glb_trun(:,1), log(glb_trun(:,2)), line_specs{i}, 'LineWidth', line_width);
end
legend(legend_name_global);
xlabel('Time/s');
ylabel('Energy(log-scale)');
fig_glb.Position = [500,500,1280,720];

%subfigure for thread
legend_name_thread = cell(nt);
for i=1:nt
    legend_name_thread{i} = sprintf('thread %d', i);
    disp(legend_name_thread{i});
end

hold off;

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize',30);
set(0, 'DefaultTextFontname', 'Times New Roman');
set(0, 'DefaultTextFontSize', 30);
line_width = 2.0;
for i=4:-2:2
    fig_thread = figure(i+1);
    hold on;
    mid = i;
    for j=1:nt
        filepath = sprintf('%s/plot_%s_thread%d.txt', dataset, method_name{mid}, j-1);
        disp(filepath);
        thd = dlmread(filepath);
        thd_trun = thd(thd(:,1) < max_time / 2, :);
        plot(thd_trun(:,1), log(thd_trun(:,2)), 'LineWidth', line_width);
    end
    if i==2
        legend('PAE Thread 1', 'PAE Thread 2', 'PAE Thread 3', 'PAE Thread 4');
    else
        legend('SM-MF(ours) Thread 1', 'SM-MF(ours) Thread 2', 'SM-MF(ours) Thread 3', 'SM-MF(ours) Thread 4');
    end
    xlabel('Time/s');
    ylabel('Energy(log-scale)');
    fig_thread.Position = [500,500,640,500];
    hold off;
end

%draw energy vs. number of threads
% figure_nthreads = figure(10);
% hold on;
% for i=2:2:8
%     filepath = sprintf('%s/SM-MF-2thread/plot_Swarn_global%d.txt', dataset, i);
%     disp(filepath);
%     m = dlmread(filepath);
%     m_trun = m(m(:,1) < max_time, :);
%     plot(m_trun(:,1), log(m_trun(:,2)), 'LineWidth', line_width);
% end
% legend('2 threads', '4 threads', '6 threads', '8 threads');
% figure_nthreads.Position=[500,500,640,500];
% hold off;
end