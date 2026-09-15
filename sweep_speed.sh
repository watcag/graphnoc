#!/usr/bin/env bash

echo "N, batch_size, limit, model_type, emb_size, num_layers, model_on_gpu, ratio_data, eval_gnn, eval_qor, total_items, time_spent, items_per_sec" > sweep_speed.csv

list_N=(3 4 5 6 7)
list_batch=(1 2 4 8 16 32 64 128 256)
num_layers=(10 15 20)
emb_size=(256 128 64)

base_args=" --model_type graph_sage_class_reg"
base_args+=" --limit 1000"
base_args+=" --dataset_folder /home/gsmalik/work/common_noc/datasets"
base_args+=" --qor_clib_path /home/gsmalik/work/hoplite-ml/codegen/exe/west_fifo_network/libwest_fifo_network.so"
base_args+=" --ratio_data 0.3"

for N in "${list_N[@]}"
do
    for batch in "${list_batch[@]}"
	do
		for test_type in "eval_gnn" "eval_qor"
		do
			args=" --N $N"
			args+=" --batch_size $batch" 
			args+=" --$test_type"
			# run gnn model
			if [ "$test_type" == "eval_gnn" ]
			then
				for index in 0 1 2
				do
					model_args="--num_layers ${num_layers[$index]} --emb_size ${emb_size[$index]}"
					# run on CPU
					python test_speed.py $base_args $args $model_args
					# run on GPU
					python test_speed.py $base_args $args $model_args --model_on_gpu
				done
			# run qor
			elif [ "$test_type" == "eval_qor" ]
			then
				python test_speed.py $base_args $args
			fi
		done
	done
done