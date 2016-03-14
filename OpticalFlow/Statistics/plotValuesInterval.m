Victor_raw_values = load('output_values_Victor');
Victor = convertValues(Victor_raw_values, 0);
solution_exchange_raw_values = load('output_values_solution_exchange');
solution_exchange = convertValues(solution_exchange_raw_values, 0);
solution_exchange_2_raw_values = load('output_values_solution_exchange_2');
solution_exchange_2 = convertValues(solution_exchange_2_raw_values, 0);
solution_exchange_5_raw_values = load('output_values_solution_exchange_5');
solution_exchange_5 = convertValues(solution_exchange_5_raw_values, 0);
solution_exchange_9_raw_values = load('output_values_solution_exchange_9');
solution_exchange_9 = convertValues(solution_exchange_9_raw_values, 0);


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
plot(solution_exchange_2(:, 1), log(solution_exchange_2(:, 2)), solution_exchange(:, 1), log(solution_exchange(:, 2)), solution_exchange_5(:, 1), log(solution_exchange_5(:, 2)), solution_exchange_9(:, 1), log(solution_exchange_9(:, 2)), Victor(:, 1), log(Victor(:, 2)), 'LineWidth', 2);
legend('k=1', 'k=2', 'k=4', 'k=8', 'k=\infty');
xlabel('Time/s');
ylabel('Energy(log-scale)');
fig.Position = [500, 500, 1280, 720];

print(fig, '../../paper/figure/optical_flow_by_interval', '-dpng');
