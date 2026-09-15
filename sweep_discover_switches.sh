#!/user/bin/env bash

echo "N, batch_size, population, trace, rate, num_layers, emb_size, device, model_type, min_wclatency, total_time" > sweep_discover_switches.csv

list_N=(3 4 5 6 7)
list_rate=(0.01 0.03 0.05 0.07 0.09)

population=512
batch_size=128
num_loops=100
num_hybrid_loops=$(( num_loops / 4 ))
path_hoplite_ml="/home/gsmalik/work/hoplite-ml"
path_common_noc="/home/gsmalik/work/common_noc"

list_num_layers=(10 15 20)
list_emb_size=(256 128 64)
list_model_load_path=(
    "${path_common_noc}/saved_models/hml_all_sizes_large/model_0.2935948669910431.pt"
    "${path_common_noc}/saved_models/hml_all_sizes_medium/model_0.35818013548851013.pt"
    "${path_common_noc}/saved_models/hml_all_sizes_small/model_0.5647122263908386.pt"
)
list_traces=(
    "random"
    "local"
    # "add20"
    # "amazon0302"
    # "bomhof_circuit_1"
    # "bomhof_circuit_2"
    # "bomhof_circuit_3"
    # "hamm_memplus"
    # "human_gene2"
    # "roadNet-CA"
    # "simucad_dac"
    # "simucad_ram2k"
    # "soc-Slashdot0902"
    # "web-Google"
    # "web-Stanford"
    # "wiki-Vote"
)


base_cmd="python test_discover_switches.py"
base_cmd+=" --qor_clib_path ${path_hoplite_ml}/codegen/exe/west_fifo_network/libwest_fifo_network.so"
base_cmd+=" --population ${population} --batch_size ${batch_size}"
base_cmd+=" --dataset_folder ${path_common_noc}/datasets"

for trace in "${list_traces[@]}"
do
    for N in "${list_N[@]}"
    do
        for rate in "${list_rate[@]}"
        do
            # run qor
            common_args="--N $N --rate $rate"
            if [ "$trace" = "local" ] || [ "$trace" = "random" ]; then
                common_args+=" --trace_path ${path_hoplite_ml}/bench/${trace}_${N}x${N}-1.dat"
            else
                common_args+=" --trace_path ${path_hoplite_ml}/bench/${trace}_${N}x${N}.dat"
            fi
            $base_cmd $common_args --eval_qor --num_loops $num_loops
            
            for index in 0 1 2
            do
                num_layers=${list_num_layers[$index]}
                emb_size=${list_emb_size[$index]}
                model_load_path=${list_model_load_path[$index]}
                
                gnn_args="--model_type graph_sage_class_reg"
                gnn_args+=" --num_layers ${num_layers}"
                gnn_args+=" --emb_size ${emb_size}"
                gnn_args+=" --model_load_path ${model_load_path}"
                
                # run GNN on GPU
                $base_cmd $common_args $gnn_args --model_on_gpu --eval_gnn --num_loops $num_loops
                
                # run hybrid on GPU
                time=$( sed -n '2p' tmp )
                best_string=$(sed -n '1p' tmp | cut -d "[" -f 2 | cut -d "]" -f 1)
                $base_cmd $common_args $gnn_args --eval_qor --model_on_gpu --init_sample $best_string --init_time $time --num_loops $num_hybrid_loops

                # run GNN on CPU
                $base_cmd $common_args $gnn_args --eval_gnn --num_loops $num_loops
                
                # run hybrid on CPU
                time=$( sed -n '2p' tmp )
                best_string=$(sed -n '1p' tmp)
                # best_string=$(sed -n '1p' tmp | cut -d "[" -f 2 | cut -d "]" -f 1)
                $base_cmd $common_args $gnn_args --eval_qor --init_sample $best_string --init_time $time --num_loops $num_hybrid_loops
            done            
        done
    done
done