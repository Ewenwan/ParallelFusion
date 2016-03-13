%solution_exchange_raw_values = load('output_values_solution_exchange');
%solution_exchange = convertValues(solution_exchange_raw_values, 0);
full_raw_values = load('output_values_full');
full= convertValues(full_raw_values, 0);
full_5_3_raw_values = load('output_values_full_5_3');
full_5_3= convertValues(full_5_3_raw_values, 0);
full_7_3_raw_values = load('output_values_full_7_3');
full_7_3= convertValues(full_7_3_raw_values, 0);
full_9_3_raw_values = load('output_values_full_9_3');
full_9_3= convertValues(full_9_3_raw_values, 0);
full_15_3_raw_values = load('output_values_full_15_3');
full_15_3= convertValues(full_15_3_raw_values, 0);

Victor_raw_values = load('output_values_Victor');
Victor = convertValues(Victor_raw_values, 1);
multiway_raw_values = load('output_values_multiway');
multiway = convertValues(multiway_raw_values, 0);
multiway_5_raw_values = load('output_values_multiway_5');
multiway_5 = convertValues(multiway_5_raw_values, 0);
multiway_7_raw_values = load('output_values_multiway_7');
multiway_7 = convertValues(multiway_7_raw_values, 0);
multiway_9_raw_values = load('output_values_multiway_9');
multiway_9 = convertValues(multiway_9_raw_values, 0);
multiway_11_raw_values = load('output_values_multiway_11');
multiway_11 = convertValues(multiway_11_raw_values, 0);

% dlmwrite('statistics_sequential.txt', sequential, '\t');
% dlmwrite('statistics_Victor.txt', Victor, '\t');
% dlmwrite('statistics_solution_exchange.txt', solution_exchange, '\t');
% dlmwrite('statistics_multiway.txt', multiway, '\t');
% dlmwrite('statistics_full.txt', full, '\t');



%plot(sequential(:, 1), sequential(:, 2), '-xm', Victor(:, 1), Victor(:,
%2), '-.c', solution_exchange(:, 1), solution_exchange(:, 2), '-*b', multiway(:, 1), multiway(:, 2), '-og', full(:, 1), full(:, 2), '-+r');\
figure(1);
%plot(full(:, 1), log(full(:, 2)), 'c', full_5_3(:, 1), log(full_5_3(:, 2)), 'b', full_7_3(:, 1), log(full_7_3(:, 2)), 'g', full_9_3(:, 1), log(full_9_3(:, 2)), 'r');
plot(Victor(:, 1), log(Victor(:, 2)), multiway(:, 1), log(multiway(:, 2)), multiway_5(:, 1), log(multiway_5(:, 2)), multiway_7(:, 1), log(multiway_7(:, 2)), multiway_9(:, 1), log(multiway_9(:, 2)), multiway_11(:, 1), log(multiway_11(:, 2)), 'LineWidth', 2);
legend('\alpha=1', '\alpha=3', '\alpha=5', '\alpha=7', '\alpha=9', '\alpha=11');
xlabel('Time/s');
ylabel('Energy');
fig.Position = [500, 500, 1280, 720];




