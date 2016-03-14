full_raw_values = load('output_values_full');
full = convertValues(full_raw_values, 0);
full_thread_1_raw_values = load('output_values_full_thread_1');
full_thread_1 = convertValues(full_thread_1_raw_values, 0);
full_thread_2_raw_values = load('output_values_full_thread_2');
full_thread_2 = convertValues(full_thread_2_raw_values, 0);
full_thread_3_raw_values = load('output_values_full_thread_3');
full_thread_3 = convertValues(full_thread_3_raw_values, 0);



% dlmwrite('statistics_sequential.txt', sequential, '\t');
% dlmwrite('statistics_Victor.txt', Victor, '\t');
% dlmwrite('statistics_solution_exchange.txt', solution_exchange, '\t');
% dlmwrite('statistics_multiway.txt', multiway, '\t');
% dlmwrite('statistics_full.txt', full, '\t');

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 30);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 30);

%plot(sequential(:, 1), sequential(:, 2), '-xm', Victor(:, 1), Victor(:,
%2), '-.c', solution_exchange(:, 1), solution_exchange(:, 2), '-*b', multiway(:, 1), multiway(:, 2), '-og', full(:, 1), full(:, 2), '-+r');\
fig = figure(1);
plot(full_thread_1(:, 1), log(full_thread_1(:, 2)), full_thread_2(:, 1), log(full_thread_2(:, 2)), full_thread_3(:, 1), log(full_thread_3(:, 2)), full(:, 1), log(full(:, 2)), 'LineWidth', 2);
legend('N=1', 'N=2', 'N=3', 'N=4');
xlabel('Time/s');
ylabel('Energy(log-scale)');
box off;
fig.Position = [500, 500, 640, 500];

print(fig, '../../paper/figure/layered_depthmap_by_N', '-dpng');
