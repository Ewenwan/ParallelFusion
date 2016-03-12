./OpticalFlow --num_threads=1 --num_iterations=200 --num_proposals_in_total=1 --num_proposals_from_others=0 --result_index=0
./OpticalFlow --num_threads=5 --num_iterations=100 --num_proposals_in_total=1 --num_proposals_from_others=0 --solution_exchange_interval=3 --use_monitor_thread=true --result_index=1
./OpticalFlow --num_threads=5 --num_iterations=100 --num_proposals_in_total=1 --num_proposals_from_others=1 --solution_exchange_interval=3 --use_monitor_thread=true --result_index=2
./OpticalFlow --num_threads=5 --num_iterations=50 --num_proposals_in_total=3 --num_proposals_from_others=0 --solution_exchange_interval=3 --use_monitor_thread=true --result_index=3
./OpticalFlow --num_threads=5 --num_iterations=50 --num_proposals_in_total=3 --num_proposals_from_others=3 --solution_exchange_interval=3 --use_monitor_thread=true --result_index=4

./OpticalFlow --num_threads=1 --num_iterations=20 --num_proposals_in_total=2 --num_proposals_from_others=0 -- --solution_exchange_interval=3 --result_index=5
./OpticalFlow --num_threads=2 --num_iterations=20 --num_proposals_in_total=2 --num_proposals_from_others=1 -- --solution_exchange_interval=3 --result_index=6
./OpticalFlow --num_threads=4 --num_iterations=20 --num_proposals_in_total=2 --num_proposals_from_others=1 -- --solution_exchange_interval=3 --result_index=7
./OpticalFlow --num_threads=8 --num_iterations=20 --num_proposals_in_total=2 --num_proposals_from_others=1 -- --solution_exchange_interval=3 --result_index=8
