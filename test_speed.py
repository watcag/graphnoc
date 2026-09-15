from models import create_models
import argparse
from data import make_loaders
import torch
import tqdm
import time
from hoplite_ml_qor import pyQoR
import pandas as pd
import numpy as np
from math import sqrt
from multiprocessing import Process


def convert_graph_to_df(data):
    pass
    # the graph already should have file name which will give us
    # info about individual src, dst trace pair. also, the graph
    # has switch modes which will give us info about switches.
    # so we already have switches (from data.switch) and traces
    # src dst (from data.filename). we only need rates now, which
    # we can get from data.x/100 (confirm the 100 part)
    df = pd.read_csv(data.trace_name, sep=r",\s+", engine="python")
    df["R"] = [data["trace"].x[0] / 100] * len(df.index)

    # TODO (@gsmalik) add a node or channel to 'trace' to capture different
    df["B"] = [1] * len(df.index)

    # get switch modes and convert them into format that qor
    # accepts
    switch_modes = np.array(
        [0 if mode == "buf" else 1 for mode in data.switch_modes], dtype=np.double
    )
    N = int(sqrt(len(switch_modes)))
    # print(f"{N=}, {switch_modes=} {df=}")

    return (N, switch_modes, df)


def benchmark_qor(dataloader, batch_size, qor_clib_path):
    # init a counter of time
    total_time = 0

    # create a list of parallel qor's func. something like below
    list_qor_func = []
    for _ in range(batch_size):
        qor_obj = pyQoR(qor_clib_path)
        qor_func = qor_obj.analyze_network
        list_qor_func.append(qor_func)

    # then we can iterate over our dataloader
    pbar = tqdm.tqdm(dataloader)
    for index, samples in enumerate(pbar):
        samples.cpu().to("cpu")
        proc_list = []
        # now iterate over each sample
        for i in range(len(samples)):
            # first we convert to df
            N, switch_modes, df = convert_graph_to_df(samples[i])
            proc = Process(target=list_qor_func[i], args=(switch_modes, N, df, 1,1))
            proc_list.append(proc)

        # now we start our counter
        time_start = time.time()

        # now we launch each process
        for proc in proc_list:
            proc.start()

        # now we wait for each proc to finish
        for proc in proc_list:
            proc.join()

        # now we get time end
        time_end = time.time()

        # now we increment total time
        total_time += time_end - time_start

        pbar.set_postfix(
            {
                "it/s": ((index + 1) * batch_size) / total_time,
                "index": index,
                "total_time": total_time,
            }
        )

    return total_time


def benchmark_gnn(gnn_model, dataloader, device):
    # init a counter of time
    total_time = 0

    # then we can iterate over our dataloader
    pbar = tqdm.tqdm(dataloader)
    for index, samples in enumerate(pbar):
        samples.to(device)

        # start time
        time_start = time.time()
        gnn_model(samples)
        time_end = time.time()

        # now we increment total time
        total_time += time_end - time_start
        pbar.set_postfix(
            {
                "it/s": ((index + 1) * batch_size) / total_time,
                "index": index,
                "total_time": total_time,
            }
        )

    return total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommonNoC entry point for training")
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Size of the NoC for which to run the program",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["graph_sage", "graph_sage_class_reg"],
        required=False,
        default="graph_sage",
        help="The type of model to use",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        required=False,
        default=5,
        help="The numbers of layers to use in the model",
    )

    parser.add_argument(
        "--emb_size",
        type=int,
        required=False,
        default=128,
        help="Size of the model's nodes' activations.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=1000,
        help="""Maximum output value that the model has to predict. Helps with
        quicker convergence""",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=128,
        help="Batch size of data. also acts as num threads for qor",
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path from which the datasets will be loaded.",
    )

    parser.add_argument(
        "--model_on_gpu",
        action="store_true",
        help="""Model is loaded onto the GPU if set. CPU if not.""",
    )

    parser.add_argument(
        "--qor_clib_path",
        type=str,
        required=True,
        help="The full path to the clib library for analysis",
    )

    parser.add_argument(
        "--ratio_data",
        type=float,
        required=False,
        default=1,
        help="""Amount of data to use. Applies to training, validation and testing
        Must be between 0 and 1""",
    )

    parser.add_argument(
        "--eval_qor",
        action="store_true",
        help="""Model is loaded onto the GPU if set. CPU if not.""",
    )

    parser.add_argument(
        "--eval_gnn",
        action="store_true",
        help="""Model is loaded onto the GPU if set. CPU if not.""",
    )

    # TODO: move this to its own function?
    args = parser.parse_args()
    N = args.N
    dataset_folder = args.dataset_folder
    batch_size = args.batch_size
    limit = args.limit
    model_type = args.model_type
    emb_size = args.emb_size
    num_layers = args.num_layers
    model_on_gpu = args.model_on_gpu
    qor_clib_path = args.qor_clib_path
    ratio_data = args.ratio_data
    eval_gnn = args.eval_gnn
    eval_qor = args.eval_qor

    assert not (eval_qor and eval_gnn)

    # check wether to use CPU or GPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and model_on_gpu) else "cpu"
    )

    # create data loaders
    train_loader, _, _ = make_loaders(
        list_N=[N],
        dataset_folder=dataset_folder,
        ratio_data=ratio_data,
        batch_size=batch_size,
        limit=limit,
        use_weighting=False,
        shuffle_train=False,
    )

    print(args)

    if eval_gnn:
        # create model
        model = create_models(
            model_type=model_type,
            emb_size=emb_size,
            num_layers=num_layers,
            dropout=0,
            limit=limit,
        )

        # convert model to hetero model. See
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        model.to_hetero(example=next(iter(train_loader)))
        # move model to device
        model.to(device)
        # put model in eval mode
        model.eval()

        time_spent = benchmark_gnn(model, train_loader, device)

    if eval_qor:
        time_spent = benchmark_qor(train_loader, batch_size, qor_clib_path)

total_items = len(train_loader)*batch_size
# write to file
with open("sweep_speed.csv", "a") as f:
    f.write(
        f"{N}, {batch_size}, {limit}, {model_type}, {emb_size}, {num_layers}, {model_on_gpu}, {ratio_data}, {eval_gnn}, {eval_qor}, {total_items}, {time_spent}, {total_items/time_spent}\n"
    )
