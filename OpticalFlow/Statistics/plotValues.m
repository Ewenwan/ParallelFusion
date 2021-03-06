sequential_raw_values = load('output_values_sequential');
sequential = convertValues(sequential_raw_values, 0);
Victor_raw_values = load('output_values_Victor');
Victor = convertValues(Victor_raw_values, 1);
solution_exchange_raw_values = load('output_values_solution_exchange_5');
solution_exchange = convertValues(solution_exchange_raw_values, 0);
multiway_raw_values = load('output_values_multiway');
multiway = convertValues(multiway_raw_values, 0);
full_raw_values = load('output_values_full');
full = convertValues(full_raw_values, 0);
hierarchy_raw_values = load('output_values_hierarchy');
hierarchy = convertValues(hierarchy_raw_values, 0);

%dlmwrite('statistics_sequential.txt', sequential, '\t');
%dlmwrite('statistics_Victor.txt', Victor, '\t');
%dlmwrite('statistics_solution_exchange.txt', solution_exchange, '\t');
%dlmwrite('statistics_multiway.txt', multiway, '\t');
%lmwrite('statistics_full.txt', full, '\t');


Victor_by_thread = {4};
for (thread = 0:3)
    thread_values = convertValues([Victor_raw_values(1, :); Victor_raw_values(find(Victor_raw_values(:, 3) == thread), :)], 0);
        Victor_by_thread{thread + 1} = thread_values;
end
%dlmwrite('thread_Victor.txt', Victor_by_thread, '\t');

solution_exchange_by_thread = {4}
for (thread = 0:3)
    thread_values = convertValues([solution_exchange_raw_values(1, :); solution_exchange_raw_values(find(solution_exchange_raw_values(:, 3) == thread), :)], 0);
        solution_exchange_by_thread{thread + 1} = thread_values;
end
%dlmwrite('thread_solution_exchange.txt', solution_exchange_by_thread, '\t');


%plot(sequential(:, 1), sequential(:, 2), '-xm', Victor(:, 1), Victor(:,
%2), '-.c', solution_exchange(:, 1), solution_exchange(:, 2), '-*b', multiway(:, 1), multiway(:, 2), '-og', full(:, 1), full(:, 2), '-+r');
%figureHandle = gcf;

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 34);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 34);

fig_1 = figure(1);
plot(sequential(:, 1), log(sequential(:, 2)), '--', Victor(:, 1), log(Victor(:, 2)), '--', hierarchy(:, 1), log(hierarchy(:, 2)), '--', solution_exchange(:, 1), log(solution_exchange(:, 2)), multiway(:, 1), log(multiway(:, 2)), full(:, 1), log(full(:, 2)), 'LineWidth', 3);
l = legend('FM', 'PFM', 'HFM', 'SF-MF(ours)', 'SF-SS(ours)', 'SF(ours)');
xlabel('Time/s');
ylabel('Energy(log-scale)');
%ylim([8, 9.8]);
box off;
fig_1.Position = [500, 500, 1280, 720];

print(fig_1, '../../paper/figure/optical_flow_convergence', '-dpng');

set(0, 'DefaultAxesFontSize', 30);
set(0, 'DefaultTextFontSize', 30);

fig_2 = figure(2);
%plot(Victor_by_thread{1}(:, 1), log(Victor_by_thread{1}(:, 2)), '-+m', Victor_by_thread{2}(:, 1), log(Victor_by_thread{2}(:, 2)), '-om', Victor_by_thread{3}(:, 1), log(Victor_by_thread{3}(:, 2)), '-*m', Victor_by_thread{4}(:, 1), log(Victor_by_thread{4}(:, 2)), '-xm', solution_exchange_by_thread{1}(:, 1), log(solution_exchange_by_thread{1}(:, 2)), '-+b', solution_exchange_by_thread{2}(:, 1), log(solution_exchange_by_thread{2}(:, 2)), '-+b', solution_exchange_by_thread{3}(:, 1), log(solution_exchange_by_thread{3}(:, 2)), '-*b', solution_exchange_by_thread{4}(:, 1), log(solution_exchange_by_thread{4}(:, 2)), '-xb', 'LineWidth', 2)
plot(Victor_by_thread{1}(:, 1), log(Victor_by_thread{1}(:, 2)), Victor_by_thread{2}(:, 1), log(Victor_by_thread{2}(:, 2)), Victor_by_thread{3}(:, 1), log(Victor_by_thread{3}(:, 2)), Victor_by_thread{4}(:, 1), log(Victor_by_thread{4}(:, 2)), 'LineWidth', 2);
legend('PFM Thread 1', 'PFM Thread 2', 'PFM Thread 3', 'PFM Thread 4');
xlabel('Time/s');
ylabel('Energy(log-scale)');
ylim([8, 9.6]);
box off;
fig_2.Position = [500, 500, 640, 500];

print(fig_2, '../../paper/figure/optical_flow_PFM_threads', '-dpng');

fig_3 = figure(3);
plot(solution_exchange_by_thread{1}(:, 1), log(solution_exchange_by_thread{1}(:, 2)), solution_exchange_by_thread{2}(:, 1), log(solution_exchange_by_thread{2}(:, 2)), solution_exchange_by_thread{3}(:, 1), log(solution_exchange_by_thread{3}(:, 2)), solution_exchange_by_thread{4}(:, 1), log(solution_exchange_by_thread{4}(:, 2)), 'LineWidth', 2);
legend('SF-MF(ours) Thread 1', 'SF-MF(ours) Thread 2', 'SF-MF(ours) Thread 3', 'SF-MF(ours) Thread 4');
xlabel('Time/s');
ylabel('Energy(log-scale)');
ylim([8, 9.6]);
box off;
fig_3.Position = [500, 500, 640, 500];

print(fig_3, '../../paper/figure/optical_flow_SF_MF_threads', '-dpng');


