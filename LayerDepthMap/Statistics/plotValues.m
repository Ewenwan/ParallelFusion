%sh readNumbers.sh

sequential_raw_values = load('output_values_sequential');
sequential = convertValues(sequential_raw_values, 0);
Victor_raw_values = load('output_values_Victor');
Victor = convertValues(Victor_raw_values, 1);
solution_exchange_raw_values = load('output_values_solution_exchange');
solution_exchange = convertValues(solution_exchange_raw_values, 0);
multiway_raw_values = load('output_values_multiway');
multiway = convertValues(multiway_raw_values, 0);
full_raw_values = load('output_values_full');
full= convertValues(full_raw_values, 0);

dlmwrite('statistics_sequential.txt', sequential, '\t');
dlmwrite('statistics_Victor.txt', Victor, '\t');
dlmwrite('statistics_solution_exchange.txt', solution_exchange, '\t');
dlmwrite('statistics_multiway.txt', multiway, '\t');
dlmwrite('statistics_full.txt', full, '\t');


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
%2), '-.c', solution_exchange(:, 1), solution_exchange(:, 2), '-*b', multiway(:, 1), multiway(:, 2), '-og', full(:, 1), full(:, 2), '-+r');\
figure(1);
plot(sequential(:, 1), log(sequential(:, 2)), 'm', Victor(:, 1), log(Victor(:, 2)), 'c', solution_exchange(:, 1), log(solution_exchange(:, 2)), 'b', multiway(:, 1), log(multiway(:, 2)), 'g', full(:, 1), log(full(:, 2)), 'r', 'LineWidth', 2);
l = legend('sequential', 'Victor', 'solution exchange', 'multiway', 'full');
xlabel('time/s');
ylabel('cost');
figure(2);
plot(Victor_by_thread{1}(:, 1), log(Victor_by_thread{1}(:, 2)), '-+m', Victor_by_thread{2}(:, 1), log(Victor_by_thread{2}(:, 2)), '-om', Victor_by_thread{3}(:, 1), log(Victor_by_thread{3}(:, 2)), '-*m', Victor_by_thread{4}(:, 1), log(Victor_by_thread{4}(:, 2)), '-xm', solution_exchange_by_thread{1}(:, 1), log(solution_exchange_by_thread{1}(:, 2)), '-+b', solution_exchange_by_thread{2}(:, 1), log(solution_exchange_by_thread{2}(:, 2)), '-+b', solution_exchange_by_thread{3}(:, 1), log(solution_exchange_by_thread{3}(:, 2)), '-*b', solution_exchange_by_thread{4}(:, 1), log(solution_exchange_by_thread{4}(:, 2)), '-xb', 'LineWidth', 2)
legend('PAE thread 1', 'PAE thread 2', 'PAE thread 3', 'PAE thread 4', 'FS-MF thread 1', 'FS-MF thread 2', 'FS-MF thread 3', 'FS-MF thread 4');
xlabel('time/s');
ylabel('cost');