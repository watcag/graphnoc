hml_all_sizes_large containts a trained noc for 3, 4, 5, 6, 7 sizes for hopliteML. This NoC was trained with 10 layers, 256 emb size, graph_sage, batch size 64 learning rate of 0.0000005. branch state is on this commit made on December 4th. you will have to see which commit added this specific section. there are two tensorboard runs. the one with Dec01\* will give you all the details regarding model perf . see command below: 

python train.py --N 3 4 5 6 7 --model_type graph_sage_class_reg --dataset_folder /home/gsmalik/work/common_noc/datasets --limit 1000 --num_layers 10 --emb_size 256 --batch_size 64 --lr 0.0000005

hml_all_sizes_medium containts a trained noc for 3, 4, 5, 6, 7 sizes for hopliteML. This NoC was trained with 15 layers, 128 emb size, graph_sage, batch size 64 learning rate of 0.0000005. branch state is on this commit made on December 19th. you will have to see which commit added this specific section. there are two tensorboard runs. the one with Dec10\* and Dec13\* will give you all the details regarding model perf . see command below: 

python train.py --N 3 4 5 6 7 --model_type graph_sage_class_reg --dataset_folder /home/gsmalik/work/common_noc/datasets --limit 1000 --num_layers 15 --emb_size 128 --batch_size 64 --lr 0.0000005

hml_all_sizes_small containts a trained noc for 3, 4, 5, 6, 7 sizes for hopliteML. This NoC was trained with 20 layers, 64 emb size, graph_sage, batch size 128 learning rate of 0.0000005. branch state is on this commit made on Feb 4th. you will have to see which commit added this specific section. there is single tensorboard run, which will give you all the details regarding model perf . see command below: 
python train.py --N 3 4 5 6 7 --model_type graph_sage_class_reg --dataset_folder /root/common_noc/datasets --limit 1000 --num_layers 20 --emb_size 64 --batch_size 128 --lr 0.0000005
