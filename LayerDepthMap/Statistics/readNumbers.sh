#!/bin/bash

cat output_sequential | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_sequential
cat output_Victor | awk '{ print $5 "\t" $8 "\t" $7}' > output_values_Victor
cat output_solution_exchange | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_solution_exchange
cat output_multiway | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_multiway
cat output_full | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full

cat output_full_2_2| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_full_2_2
cat output_full_4_3 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full_4_3
cat output_full_5_3 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full_5_3
cat output_full_7_3 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full_7_3
cat output_full_9_3 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full_9_3
cat output_full_15_3 | awk '{ print $5 "\t" $8 "\t" $7 }' > output_values_full_15_3

cat output_multiway_5| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_multiway_5
cat output_multiway_7| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_multiway_7
cat output_multiway_9| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_multiway_9
cat output_multiway_11| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_multiway_11

cat output_full_thread_1| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_full_thread_1
cat output_full_thread_2| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_full_thread_2
cat output_full_thread_3| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_full_thread_3
cat output_full_thread_8| awk '{ print $5 "\t" $8 "\t" $7}' > output_values_full_thread_8
