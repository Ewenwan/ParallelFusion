#!/usr/bin/env zsh

exe=/home/erik/Projects/ParallelFusion/build/LayerDepthMap/LayerDepthMap
data_set=/home/erik/Projects/ParallelFusion/LayerDepthMap/Inputs/cse013

data_file=raw-time-v-energy.csv


run_test() {
  run_name=$1
  alpha=$2
  beta=$3
  gamma=$4

  for i in {1..3}; do
    ${exe} -scene_name=${data_set} -num_threads=4 -num_proposals_in_total=${alpha} -num_proposals_from_others=${beta} -solution_exchange_interval=${gamma}
    cp ${data_file} "${run_name}_${i}.csv"
  done

}

# echo "Running fusion move"
# for i in {1..3}; do
#   ${exe} -scene_name=${data_set} -num_threads=1 -num_proposals_in_total=1 -num_proposals_from_others=0 -solution_exchange_interval=1
#   cp ${data_file} "fusion_move_${i}.csv"
# done

# echo "Running parallel fusion move"
# run_test "parallel_fusion_move" 1 0 1

echo "Running SF-MF"
run_test "sf-mf" 1 1 5

# echo "Running SF-SS"
# run_test "sf-ss" 3 0 1

echo "Running SF"
run_test "sf" 3 3 5


echo "Running SF-Baeysian"
run_test "bopt" 6 3 9

echo "Running SF-Grid"
run_test "grid" 10 2 7
