function [ time_energy_values ] = convertValues( raw_values, use_final_fusion )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

start_time = raw_values(1, 1);
previous_energy = inf;
%time_energy_values = [0, previous_energy]
for i = 2:size(raw_values, 1)
    if (raw_values(i, 3) == 4)
        if (use_final_fusion == 1)
            fusion_time_energy_value = raw_values(i, :);    
        end
        continue;
    end
    if (raw_values(i, 2) <= previous_energy)
        if (i == 2)
            time_energy_values = [raw_values(i, 1) - start_time, raw_values(i, 2), raw_values(i, 3)];
        else
            time_energy_values = [time_energy_values; raw_values(i, 1) - start_time, raw_values(i, 2), raw_values(i, 3)];
        end
        previous_energy = raw_values(i, 2);
    end
    if (raw_values(i, 1) > start_time + 420)
        break;
    end
end
if (use_final_fusion == 1)
    fusion_time_energy_value(1) = fusion_time_energy_value(1) - start_time;
    time_energy_values = [time_energy_values; fusion_time_energy_value];
end

end