#!/bin/bash

cat output_sequential | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_sequential
cat output_Victor | awk '{ print $5 "\t" $8 "\t" $7}' > output_values_Victor
cat output_solution_exchange | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange
cat output_multiway | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_multiway
cat output_full | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full

cat output_solution_exchange_2 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_2
cat output_solution_exchange_4 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_4
cat output_solution_exchange_5 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_5
cat output_solution_exchange_9 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_9

cat output_solution_exchange_1_other | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_1_other
cat output_solution_exchange_2_other | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_2_other
cat output_solution_exchange_3_other | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange_3_other
