# things we will need:
# 1. qor hookup
# 2. hookup to a saved gnn model
# 3. our MLE algorithm hookup

# we will set up args. make sure we will pass which qor to use (suite
# or gnn small medium large) here so we only optimise it using that.
# i say this becuase in the parent bash script, we will have running
# for qor, small, medium and large gnn.

# then we will load up a trace for which we want to optimise. we will
# keep traces as something that is passed as arg so that we can have
# an external script to control all the passing etc.

# then we will load up our GNN model or qor and pass this to MLE when
# setting up MLE

# then we will set up our MLE thingy. we can just write a new MLE thingy
# class in this file itself so that it becomes easier to make the hookups
import numpy as np
from models import create_models
from data import make_loaders

import time
import gnn
import math
from torch_geometric.loader import DataLoader
import sys
import argparse
import torch
import hoplite_ml_qor
import pandas as pd
import tqdm
from multiprocessing import Process, Manager
import gc


torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)
# TODO: also need to make sure that current_time works as intended

# TODO: we also need to have a way to parallelize for qor. if qor ends
# up being faster, you can just launch parallel runs of different NoC
# optimizations. the idea here is that GPU might get underutilised
# over a single loop (generate->evaluate->update). but if we have
# multiple loops going on in parallel (for different NoC, trace pairs),
# we can utilize the GPU properly. then we should be able to get the
# same speedups as we saw in our performance benchmarking (upto 55x)

# TODO: we also should have a way of initializing samples.


class MLE:
    def __init__(
        self,
        N,
        population,
        p,
        trace_name,
        rate_user,
        mode,
        qor_clib_path,
        num_loops=100,
        init_sample=None,
        batch_size=128,  # used by both gnn and qor.
        gnn_model=None,
        gnn_buf_id=None,
        gnn_bp_id=None,
        gnn_port_id_mappings=None,
        device="cpu",
    ):
        assert mode in ["gnn", "qor"]

        self.N = N
        self.population = population
        self.p = np.array([p] * N * N)
        self.trace_name = trace_name
        # TODO: make sure usage of rate between GNN and QoR is same
        self.rate_user = rate_user
        self.mode = mode
        self.batch_size = batch_size
        self.init_sample = init_sample
        self.use_init_sample = self.init_sample is not None
        self.num_loops = num_loops

        self.qor_clib_path = qor_clib_path

        self.gnn_model = gnn_model

        if mode == "gnn":
            assert gnn_buf_id is not None
            assert gnn_bp_id is not None
            assert gnn_port_id_mappings is not None

            self.gnn_buf_id = gnn_buf_id
            self.gnn_bp_id = gnn_bp_id
            self.gnn_port_id_mappings = gnn_port_id_mappings

            self.gnn_buf_bp_mapping = {0: self.gnn_buf_id, 1: self.gnn_bp_id}

            self.hl_grapher = gnn.HopliteGrapher(
                buf_id=self.gnn_buf_id,
                bp_id=self.gnn_bp_id,
                port_id_mapping=self.gnn_port_id_mappings,
                qor_clib_path=self.qor_clib_path,
            )

            self.device = device

            samples = self.generate_samples()
            self.dataloader = self.encode_noc_to_gnn(samples)

        self.best_fitness = sys.float_info.max

    def loop(self):
        current_time = 0
        # while not ((self.p == 0.0) | (self.p == 1.0)).all():
        for _ in tqdm.tqdm(range(self.num_loops)):
            # generate samples
            samples = self.generate_samples()
            # print(samples)

            # get fitness
            if self.gnn_model:
                fitness, current_time = self.get_gnn_prediction(samples, current_time)
            else:
                fitness, current_time = self.get_qor_prediction(samples, current_time)

            # update
            self.update(samples, fitness)

        temp_qor_tool = hoplite_ml_qor.pyQoR(self.qor_clib_path)
        df = pd.read_csv(self.trace_name, sep=",\s+", engine="python")
        temp_qor_tool.analyze_network(
            x=self.best,
            N=self.N,
            df=df,
            burst_user=1,
            rate_user=self.rate_user,
        )
        actual_best_fitness = (
            temp_qor_tool.wclatency if not temp_qor_tool.network_unstable else 100000000
        )
        print(
            f"Best: {self.best_fitness=} {self.best=} {actual_best_fitness} {current_time=}"
        )
        return self.best, actual_best_fitness, current_time

    def generate_samples(self):
        """
        Generates sample candidates with different switch configuration on a per
        switch basis.

        Returns
        -------
        samples: np.ndarray of shape (population, N*N)
        """

        samples = np.zeros((self.N * self.N, self.population))
        for index in range(self.N * self.N):
            samples[index] = np.random.binomial(1, self.p[index], (1, self.population))
            samples[index][0] = 0
            samples[index][1] = 1
        samples = samples.T

        if self.use_init_sample:
            for i in range(int(0.25 * self.population)):
                samples[i] = self.init_sample
            self.use_init_sample = False
        return samples

    def update(self, samples, fitness):
        """
        Updates values of the binomial distribution variables.

        Parameters
        ----------
        samples: np.ndarray
            The candidates for which wclatency needs to be calculated
        fitness: list
            Worst case latencies of each candidate in `samples`

        Returns
        -------
        None. Updates self.p to reflect newest fitness.
        """
        fitness = np.array(fitness)

        assert np.size(fitness) == self.population

        # for sample, fit in zip(samples, fitness):
        #     print(sample, fit)

        # get top 25% fittest candidates
        top_sel = int(0.25 * self.population)
        fittest = samples[np.argsort(fitness)][:top_sel]
        current_best_fitness = fitness[np.argsort(fitness)][0]

        # update global fittest if applicable
        if current_best_fitness <= self.best_fitness:
            self.best = fittest[0]
            self.best_fitness = current_best_fitness

        # update proababilities
        self.p = np.mean(fittest, axis=0)

    def run_qor(
        self, index, results_dict, qor_tool, sample, N, df, burst_user, rate_user
    ):
        results_dict[index] = qor_tool.analyze_network(
            x=sample, N=N, df=df, burst_user=burst_user, rate_user=rate_user
        )

    def get_qor_prediction(self, samples, current_time):
        """
        Gets worst case latency using Hoplite Analysis tool.

        Parameters
        ----------
        qor_tool: hoplite_ml_qor.pyQOR
            Tool to use to calculate wclatency
        samples: np.ndarray
            The candidates for which wclatency needs to be calculated
        current_time: float
            A counter of how much time has been spent

        Returns
        -------
        resuls: list
            Worst case latencies for each candidate in `samples`
        current_time: float
            Updated counter of time spent analyzing candidates in `samples`.
        """
        results = []
        df = pd.read_csv(self.trace_name, sep=",\s+", engine="python")

        # init multiprocessing
        proc_list = []  # to hold `Process` processes
        size = 0  # to keep track of batch size

        # each parallel process gets its own QoR tool
        qor_list = [
            hoplite_ml_qor.pyQoR(self.qor_clib_path) for _ in range(self.batch_size)
        ]

        # this is where results will be updated. index in batch size will be used
        # as key of the dict.
        results_dict = Manager().dict()
        for sample in samples:
            # we keep adding to parallel process list until we have a batch_size
            # sized process list that we can launch in parallel
            if size < self.batch_size:
                proc = Process(
                    target=self.run_qor,
                    args=(
                        size,
                        results_dict,
                        qor_list[size],
                        sample,
                        self.N,
                        df,
                        1,
                        self.rate_user,
                    ),
                )
                proc_list.append((proc, qor_list[size]))
                size += 1

            # we launch all processes in parallel if we have a batch_size sized
            # process list.
            if size == self.batch_size:
                start_time = time.time()

                # now we launch each process
                for proc, _ in proc_list:
                    proc.start()

                # now we wait for each proc to finish
                for proc, _ in proc_list:
                    proc.join()

                # now we get time end
                end_time = time.time()

                current_time += end_time - start_time

                for index in range(self.batch_size):
                    results.append(results_dict[index])

                # we init parallel processes for next mini batch
                proc_list = []
                size = 0
                qor_list = [
                    hoplite_ml_qor.pyQoR(self.qor_clib_path)
                    for _ in range(self.batch_size)
                ]
                results_dict = Manager().dict()

        return results, current_time

    def get_gnn_prediction(self, samples, current_time):
        """
        Gets prediction of worst case latency using GNN model.

        Parameters
        ----------
        samples: np.ndarray
            The candidates for which wclatency needs to be calculated
        current_time: float
            A counter of how much time has been spent

        Returns
        -------
        resuls: list
            Worst case latencies for each candidate in `samples`
        current_time: float
            Updated counter of time spent analyzing candidates in `samples`.
        """
        transformed_samples = torch.tensor(
            samples, dtype=torch.float, device=self.device
        ).flatten()[:, None]

        results_list = []
        for index, data_sample in enumerate(self.dataloader):
            # inplace update
            start_index = index * self.batch_size * self.N * self.N
            end_index = (index + 1) * self.batch_size * self.N * self.N
            data_sample["switch"]["x"] = transformed_samples[start_index:end_index]

            # move to device
            data_sample.to(self.device)

            # get prediction from GNN
            start_time = time.time()
            result_data_sample, _, _ = self.gnn_model.get_prediction(data_sample)
            end_time = time.time()
            current_time += end_time - start_time

            results_list.append(result_data_sample.detach().cpu())

        results = torch.cat(results_list, dim=0)
        return results, current_time

    def encode_noc_to_gnn(self, samples):
        """
        Encode NoC candidate and trace into a GNN dataloader

        Parameters
        ----------
        samples: np.ndarray
            The candidates for which wclatency needs to be calculated

        Returns
        -------
        dataloader: torch.geometric.loader.DataLoader
            The dataloader that holds all NoC candidates and traces
            encoded as individual graph representations.
        """
        data = []
        for sample in samples:
            switch_modes = [
                "buf" if self.gnn_buf_bp_mapping[switch] == self.gnn_buf_id else "bp"
                for switch in sample
            ]
            # convert into GNN represenation and append
            data.append(
                self.hl_grapher.make_hl_ml_gnn(
                    N=self.N,
                    switch_modes=switch_modes,
                    trace_file=self.trace_name,
                    rate_user=self.rate_user,
                    burst_user=1,
                )
            )

        # convert it into data loader
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False)

        return dataloader


def run_qor_only(
    N,
    population,
    trace_path,
    rate_user,
    batch_size,
    num_loops,
    seed=0,
    init_sample=None,
):
    print("Running QoR only")

    # set same seed
    np.random.seed(seed)

    # create MLE object
    mle = MLE(
        N=N,
        population=population,
        p=0.5,
        mode="qor",
        trace_name=trace_path,
        rate_user=rate_user,
        batch_size=batch_size,
        qor_clib_path=qor_clib_path,
        num_loops=num_loops,
        init_sample=init_sample,
    )

    # loop it
    _, min_wclatency, total_time = mle.loop()

    return min_wclatency, total_time


def run_gnn_only(
    N,
    population,
    trace_path,
    rate_user,
    batch_size,
    model,
    gnn_buf_id,
    gnn_bp_id,
    gnn_port_id_mappings,
    qor_clib_path,
    device,
    num_loops,
    seed=0,
):
    print("Running GNN only")

    # set same seed
    np.random.seed(seed)

    # create MLE object
    mle = MLE(
        N=N,
        population=population,
        p=0.5,
        mode="gnn",
        trace_name=trace_path,
        rate_user=rate_user,
        gnn_model=model,
        gnn_buf_id=gnn_buf_id,
        gnn_bp_id=gnn_bp_id,
        gnn_port_id_mappings=gnn_port_id_mappings,
        batch_size=batch_size,
        qor_clib_path=qor_clib_path,
        device=device,
        num_loops=num_loops,
    )

    best_gnn, min_wclatency_gnn, total_time_gnn = mle.loop()

    return (
        best_gnn,
        min_wclatency_gnn,
        total_time_gnn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CommonNoC entry point for testing discovery of switches"
    )
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
        help="The numbers of layers to use in the model. Only applicable to GNN models.",
    )

    parser.add_argument(
        "--emb_size",
        type=int,
        required=False,
        default=128,
        help="Size of the model's nodes' activations. Only applicable to GNN models.",
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
        "--num_loops",
        type=int,
        required=False,
        default=100,
        help="Number of loops to run MLE",
    )

    parser.add_argument(
        "--population",
        type=int,
        required=False,
        default=100,
        help="Number of candidates generated by MLE in each iteration",
    )

    parser.add_argument(
        "--qor_clib_path",
        type=str,
        required=True,
        help="The full path to the clib library for analysis.",
    )

    parser.add_argument(
        "--trace_path",
        type=str,
        required=True,
        help="The full path to the traffic trace for which switches are being discovered.",
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path from which the datasets will be loaded. Required for GNN hetero conversion",
    )

    parser.add_argument(
        "--model_load_path",
        type=str,
        required=False,
        help="""If a model checkpoint will be loaded. Program will use this model
        to run.""",
    )

    parser.add_argument(
        "--buf_id",
        type=int,
        required=False,
        default=1,
        help="The ID for representing the core HopliteBuf switch.",
    )

    parser.add_argument(
        "--bp_id",
        type=int,
        required=False,
        default=2,
        help="The ID for representing the core HopliteBP switch.",
    )

    parser.add_argument(
        "--north_id",
        type=int,
        required=False,
        default=3,
        help="Port ID of the north port of the Hoplite switch.",
    )

    parser.add_argument(
        "--west_id",
        type=int,
        required=False,
        default=4,
        help="Port ID of the north port of the Hoplite switch.",
    )

    parser.add_argument(
        "--south_id",
        type=int,
        required=False,
        default=5,
        help="Port ID of the north port of the Hoplite switch.",
    )

    parser.add_argument(
        "--east_id",
        type=int,
        required=False,
        default=6,
        help="Port ID of the north port of the Hoplite switch.",
    )

    parser.add_argument(
        "--pe_id",
        type=int,
        required=False,
        default=7,
        help="Port ID of the PE port of the Hoplite switch.",
    )

    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Rate at whcih traffic will be injected into NoC",
    )

    parser.add_argument(
        "--model_on_gpu",
        action="store_true",
        help="""Model is loaded onto the GPU if set. CPU if not.""",
    )

    parser.add_argument(
        "--file_to_write_results",
        type=str,
        required=False,
        default="sweep_discover_switches.csv",
        help="""Where results should be written.""",
    )
    parser.add_argument(
        "--eval_qor",
        action="store_true",
        help="""Wether to use qor or not""",
    )

    parser.add_argument(
        "--eval_gnn",
        action="store_true",
        help="""Wether to use GNN""",
    )

    parser.add_argument(
        "--init_sample",
        type=int,
        nargs="+",
        required=False,
        help="""Initial sample to initialize qor""",
    )

    parser.add_argument(
        "--init_time",
        type=float,
        required=False,
        help="""Initial time to add to qor""",
    )

    # TODO: move this to its own function?
    args = parser.parse_args()

    N = args.N
    batch_size = args.batch_size
    population = args.population
    limit = args.limit
    model_type = args.model_type
    emb_size = args.emb_size
    num_layers = args.num_layers
    qor_clib_path = args.qor_clib_path
    trace_path = args.trace_path
    rate = args.rate
    dataset_folder = args.dataset_folder
    model_load_path = args.model_load_path
    buf_id = args.buf_id
    bp_id = args.bp_id
    port_id_mappings = {
        "north": args.north_id,
        "west": args.west_id,
        "south": args.south_id,
        "east": args.east_id,
        "pe": args.pe_id,
    }
    model_on_gpu = args.model_on_gpu
    file_to_write_results = args.file_to_write_results
    eval_gnn = args.eval_gnn
    eval_qor = args.eval_qor
    num_loops = args.num_loops
    init_sample = args.init_sample
    init_time = args.init_time

    assert (init_sample is not None) == (init_time is not None), f"{init_sample=} {init_time=} {init_sample is not None} {init_time is not None}"

    assert not (eval_qor and eval_gnn)
    assert population % batch_size == 0

    # run QoR
    if eval_qor:
        min_wclatency, total_time = run_qor_only(
            N=N,
            population=population,
            trace_path=trace_path,
            rate_user=rate,
            batch_size=batch_size,
            num_loops=num_loops,
            init_sample=init_sample,
        )
        if init_sample and init_time:
            mode = f"{model_type}_hybrid"
            device = torch.device(
                "cuda" if (torch.cuda.is_available() and model_on_gpu) else "cpu"
            )
            total_time += init_time
        else:
            mode = "qor"
            device = "cpu"
            num_layers=0
            emb_size=0


        with open(file_to_write_results, "a") as f:
            f.write(
                f"{N}, {batch_size}, {population}, {trace_path}, {rate}, {num_layers}, {emb_size}, {device}, {mode}, {min_wclatency}, {total_time}\n"
            )

    if eval_gnn:
        assert model_load_path
        # check wether to use CPU or GPU
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and model_on_gpu) else "cpu"
        )

        # run for GNN first

        # create dataset for hetero GNN conversion
        train_loader, _, _ = make_loaders(
            list_N=[N],
            dataset_folder=dataset_folder,
            batch_size=batch_size,
            limit=limit,
            use_weighting=False,
        )

        # load up gnn model
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

        # feed a sample to init the model shapes
        model(next(iter(train_loader)).to(device))
        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # run GNN
        best_gnn, min_wclatency_gnn, total_time_gnn = run_gnn_only(
            N=N,
            population=population,
            trace_path=trace_path,
            batch_size=batch_size,
            model=model,
            gnn_buf_id=buf_id,
            gnn_bp_id=bp_id,
            gnn_port_id_mappings=port_id_mappings,
            rate_user=rate,
            qor_clib_path=qor_clib_path,
            device=device,
            num_loops=num_loops,
            seed=0,
        )

        with open(file_to_write_results, "a") as f:
            f.write(
                f"{N}, {batch_size}, {population}, {trace_path}, {rate}, {num_layers}, {emb_size}, {device}, {model_type}, {min_wclatency_gnn}, {total_time_gnn}\n"
            )

        best_string = " ".join([str(int(x)) for x in best_gnn])

        with open("tmp", "w") as f:
            f.write(f"{best_string}\n")
            f.write(f"{total_time_gnn}\n")

        # # return min_wclatency_gnn, best_gnn, total_time_gnn

        # # del globals()["f"]
        # # del globals()["min_wclatency_gnn"]
        # del globals()["train_loader"]
        # del globals()["torch"]
        # del globals()["_"]
        # del globals()["make_loaders"]
        # del globals()["model"]
        # del globals()["create_models"]
        # del globals()["checkpoint"]
        # del globals()["gnn"]
        # del globals()["DataLoader"]
        # del globals()["port_id_mappings"]
        # del globals()["buf_id"]
        # del globals()["bp_id"]
        # del globals()["model_load_path"]
        # del globals()["dataset_folder"]
        # del globals()['run_gnn_only']

        # gc.collect()
        # print(globals().keys())

        # min_wclatency_hybrid, total_time_hybrid = run_qor_only(
        #     N=N,
        #     population=population,
        #     trace_path=trace_path,
        #     rate_user=rate,
        #     batch_size=batch_size,
        #     num_loops=num_loops//2,
        #     # init_sample=best_gnn
        # )

        # with open(file_to_write_results, "a") as f:
        #     f.write(
        #         f"{N}, {batch_size}, {population}, {trace_path}, {rate}, {num_layers}, {emb_size}, {device}, hybrid_{model_type}, {min_wclatency_hybrid}, {total_time_hybrid+total_time_gnn}\n"
        #     )
