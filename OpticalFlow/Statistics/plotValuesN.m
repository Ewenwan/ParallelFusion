full_raw_values = load('output_values_full');
full = convertValues(full_raw_values, 0);
%thread_1_raw_values = load('output_values_full_thread_1');
%thread_1 = convertValues(full_thread_1_raw_values, 0);
thread_2_raw_values = load('output_values_thread_2');
thread_2 = convertValues(thread_2_raw_values, 0);
solution_exchange_raw_values = load('output_values_solution_exchange');
solution_exchange = convertValues(solution_exchange_raw_values, 0);
thread_8_raw_values = load('output_values_thread_8');
thread_8 = convertValues(thread_8_raw_values, 0);



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
plot(thread_2(:, 1), log(thread_2(:, 2)), thread_2(:, 1), log(thread_2(:, 2)), solution_exchange(:, 1), log(solution_exchange(:, 2)), thread_8(:, 1), log(thread_8(:, 2)), 'LineWidth', 2);
legend('N=1', 'N=2', 'N=4', 'N=8');
xlabel('Time/s');
ylabel('Energy(log-scale)');
fig.Position = [500, 500, 1280, 720];

print(fig, '../../paper/figure/optical_flow_by_N', '-dpng');
