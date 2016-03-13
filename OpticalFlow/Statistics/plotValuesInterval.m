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



%plot(sequential(:, 1), sequential(:, 2), '-xm', Victor(:, 1), Victor(:,
%2), '-.c', solution_exchange(:, 1), solution_exchange(:, 2), '-*b', multiway(:, 1), multiway(:, 2), '-og', full(:, 1), full(:, 2), '-+r');\
figure(1);
plot(Victor(:, 1), log(Victor(:, 2)), 'm', solution_exchange_2(:, 1), log(solution_exchange_2(:, 2)), 'c', solution_exchange(:, 1), log(solution_exchange(:, 2)), 'b', solution_exchange_5(:, 1), log(solution_exchange_5(:, 2)), 'g', solution_exchange_9(:, 1), log(solution_exchange_9(:, 2)), 'r');
legend('0', '2', '3', '5', '9');
xlabel('time/s');
ylabel('cost');


