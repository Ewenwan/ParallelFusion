function [ time_energy_values ] = convertValues( raw_values )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

start_time = raw_values(1, 1);
previous_energy = inf;
%time_energy_values = [0, previous_energy]
for i = 2:size(raw_values, 1)
    if (raw_values(i, 2) <= previous_energy)
        if (i == 2)
            time_energy_values = [raw_values(i, 1) - start_time, raw_values(i, 2)];
        else
            time_energy_values = [time_energy_values; raw_values(i, 1) - start_time, raw_values(i, 2)];
        end
        previous_energy = raw_values(i, 2);
    end
end

end