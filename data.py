import argparse
import gnn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import torch
import tqdm
from torch.utils.data import WeightedRandomSampler
import os
import shutil
from torch.utils.data import Dataset, ConcatDataset
import tqdm


class CommonNoCDataset(Dataset):
    def __init__(self, dataset_folder, N, mode, limit, ratio_data):
        self.dataset_folder = dataset_folder
        self.N = N
        self.mode = mode
        self.limit = limit

        # unzip if needed
        if not os.path.isdir(f"{dataset_folder}/{mode}_{N}"):
            print("Unzipping dataset")
            shutil.unpack_archive(
                f"{dataset_folder}/{mode}_{N}.zip", f"{dataset_folder}/{mode}_{N}"
            )

        if not os.path.isfile(
            f"{dataset_folder}/{mode}_{N}/train_filenames_{limit}.pt"
        ):
            print(f"Caching subset for limit {limit}")
            # load up list of files
            all_train_filenames = torch.load(
                f"{dataset_folder}/{mode}_{N}/list_objects.pt"
            )

            train_filenames = []
            for filename in tqdm.tqdm(all_train_filenames):
                file_path = f"{self.dataset_folder}/{self.mode}_{self.N}/{filename}"
                sample = torch.load(file_path)
                if sample.y <= self.limit:
                    train_filenames.append(filename)

            del all_train_filenames
            torch.save(
                train_filenames,
                f"{dataset_folder}/{mode}_{N}/train_filenames_{limit}.pt",
            )

        self.train_filenames = torch.load(
            f"{dataset_folder}/{mode}_{N}/train_filenames_{limit}.pt"
        )
        self.train_filenames = self.train_filenames[
            : int(ratio_data * len(self.train_filenames))
        ]

    def __len__(self):
        return len(self.train_filenames)

    def __getitem__(self, idx):

        # # !: do i need to have torch.is_tensor check here?
        file_path = (
            f"{self.dataset_folder}/{self.mode}_{self.N}/{self.train_filenames[idx]}"
        )
        sample = torch.load(file_path)
        return sample

# TODO: i think that the port id mappings of t and pi should be completely 
# different. think about it. even dl and dr have different muxes when it comes
# to t and pi. for dl, t will have to mux u and dr, whereas pi will have to
# mux dr, ur and ul.
# so what we might want to do is have a t_port_id_mapping and a pi_port_id_mapping
# and then have a hl_port_id_mapping and have assert statements:
# 1. make sure that all 3 are provided
# 2. all the values are unique among these mappings as well. that is:
#  len(unique(hl+t+pi)) == len( hl+t+pi)
def create_bft_dataset(N, pi_id, t_id, port_id_mapping, len_dataset, bft_path):
    dataset = []

    bft3_grapher = bft_gnn.BFT3Grapher(pi_id, t_id, port_id_mapping, bft_path=bft_path)
    # we reuse everything in port id mapping. just we call it u instead of ul and
    # dont use ur
    port_id_mapping_bft0 = {
        "dl": port_id_mapping["dl"],
        "dr": port_id_mapping["dr"],
        "u": port_id_mapping["ul"],
    }
    bft0_grapher = bft_gnn.BFT0Grapher(t_id, port_id_mapping_bft0, bft_path=bft_path)

    for trace_type in [
        "google",
        "random",
        "local",
        "add20",
        "amazon",
        "bomhof_1",
        "bomhof_2",
        "bomhof_3",
        "hamm",
        "human",
        "roadnet",
        "simucad_dac",
        "simucad_ram2k",
        "soc",
        "stanford",
        "wiki",
    ]:
        if len(dataset) >= len_dataset:
            break
        is_synthetic = trace_type in ["local", "random"]
        for trace_num in range(1, 25) if is_synthetic else range(1, 2):
            if len(dataset) >= len_dataset:
                break
            for bft_type in ["bft3", "bft0"]:
                grapher = bft3_grapher if bft_type == "bft3" else bft0_grapher

                if len(dataset) >= len_dataset:
                    break

                for rate_user in range(10, 100, 5):
                    print(N, trace_type, trace_num, rate_user, bft_type, len(dataset))
                    if len(dataset) >= len_dataset:
                        break
                    id = f"test/{N}_{trace_type}_{trace_num}_{rate_user}_{bft_type}"
                    data = grapher.make_gnn(N, trace_type, rate_user, id)
                    data.validate()
                    print(data.y)

                    dataset.append(data)

    return dataset


def create_dataset(
    N,
    buf_id,
    bp_id,
    port_id_mapping,
    qor_clib_path,
    traces_dir_path,
    train_val_test_split,
    dataset_folder,
    seed=0,
):
    """
    Creates dataset for hoplite ML. The ground truth is calculated using the
    Hoplite QoR analysis suite.

    Parameters
    ----------
    N: int
        Size of the NoC where the NoC is represented as N x N
    buf_id: int
        The ID for representing the core HopliteBuf switch.
    bp_id: int
        The ID for representing the core HopliteBP switch.
    port_id_mappings: dict
        The mapping of a hoplite ports to an `int` ID. Must contain 'north',
        'west', 'east', 'south' and 'pe' as keys.
    qor_clib_path: str
        The full path to the clib library for analysis
    traces_dir_path: str
        The full path to the directory containing all the traces.
        For example, if the full path to a particular trace is
        "/foo/bar/bench/trace.dat", then `traces_dir_path` will be
        "/foo/bar/bench". All traces must lie in this directory path.
    train_val_test_split: list
        A list of length 3, specifying how much of the total dataset will
        go towards train, validation and test split. Index 0 of the list
        is for train, index 1 for validation and index 2 is for test.
    dataset_folder: str
        Path to which the dataset will be saved.
    seed: int, optional
        Seed using which the random permutations of the function are
        controlled. Defaults to 0 if not specified.

    Returns
    -------
    None. 3 files are saved in the `dataset_folder` specified with the naming
    convention of train_{N}.pt, val_{N}.pt and test_{N}.pt
    """
    assert sum(train_val_test_split) == 1, "train, validation and test must add up to 1"

    os.makedirs(f"{dataset_folder}/{N}", exist_ok=False)

    # set seeds
    random.seed(seed)
    np.random.seed(seed)

    hl_grapher = gnn.HopliteGrapher(
        buf_id=buf_id,
        bp_id=bp_id,
        port_id_mapping=port_id_mapping,
        qor_clib_path=qor_clib_path,
    )

    train_dataset, val_dataset, test_dataset = [], [], []

    for trace_type in [
        "random",
        "local",
        "add20",
        "amazon0302",
        "bomhof_circuit_1",
        "bomhof_circuit_2",
        "bomhof_circuit_3",
        "hamm_memplus",
        "human_gene2",
        "roadNet-CA",
        "simucad_dac",
        "simucad_ram2k",
        "soc-Slashdot0902",
        "web-Google",
        "web-Stanford",
        "wiki-Vote" "",
    ]:
        is_synthetic = trace_type in ["local", "random"]

        # we use up more traces if the trace type is synthetic
        for trace_num in range(1, 2) if is_synthetic else range(1, 2):
            print(f"Generating {trace_type} for NoC of size {N}")
            # we generate 500 different combinations of hoplite buf/bp mix,
            # along with a pure vanilla buf and bp
            num_comb = 500
            for comb_index in tqdm.tqdm(range(num_comb + 2)):
                # create a random switch distribution or vanilla
                if comb_index < num_comb:  # random mixture of switches
                    switch_modes = [
                        "buf" if num == 0 else "bp"
                        for num in np.random.randint(0, 2, N * N)
                    ]
                elif comb_index == num_comb:  # vanilla hoplite buf
                    switch_modes = ["buf" for _ in range(N * N)]
                else:  # vanilla hoplite bp
                    switch_modes = ["bp" for _ in range(N * N)]

                # loop over different rates
                dataset = []
                for rate_user in np.arange(0.001, 0.25, 0.01):
                    # determine name of trace
                    trace_suffix = f"{traces_dir_path}/{trace_type}_{N}x{N}"
                    if is_synthetic:
                        trace_name = f"{trace_suffix}-{trace_num}.dat"
                    else:
                        trace_name = f"{trace_suffix}.dat"

                    # create data
                    data = hl_grapher.make_hl_ml_gnn(
                        N=N,
                        switch_modes=switch_modes,
                        trace_file=trace_name,
                        rate_user=rate_user,
                        burst_user=1,
                    )
                    data.validate()
                    data.trace_name = trace_name
                    data.switch_modes = switch_modes
                    # add to data if feasible.
                    if not data.network_unstable:
                        # dataset.append(data)
                        filename = (
                            f"{N}_{trace_type}_{trace_num}_{comb_index}_{rate_user}.pt"
                        )
                        dataset.append(filename)
                        torch.save(data, f"{dataset_folder}/{N}/{filename}")
                    else:
                        break

                # split dataset into train, val and test according to specified
                # ratios.
                random.shuffle(dataset)
                len_dataset = len(dataset)
                train_index = int(train_val_test_split[0] * len_dataset)
                val_index = int(train_val_test_split[1] * len_dataset)

                train_dataset += dataset[:train_index]
                val_dataset += dataset[train_index : train_index + val_index]
                test_dataset += dataset[train_index + val_index :]

    move_files_and_package_as_zip(N, dataset_folder, "train", train_dataset)
    move_files_and_package_as_zip(N, dataset_folder, "val", val_dataset)
    move_files_and_package_as_zip(N, dataset_folder, "test", test_dataset)

    shutil.rmtree(f"{dataset_folder}/{N}")


def move_files_and_package_as_zip(N, dataset_folder, name, dataset_filenames):
    os.makedirs(f"{dataset_folder}/{N}/{name}", exist_ok=False)

    for filename in dataset_filenames:
        shutil.move(
            f"{dataset_folder}/{N}/{filename}",
            f"{dataset_folder}/{N}/{name}/{filename}",
        )

    # TODO: need to add a txt file or maybe save up the list of files as pt as well.
    torch.save(dataset_filenames, f"{dataset_folder}/{N}/{name}/list_objects.pt")

    # create a zip
    shutil.make_archive(
        f"{dataset_folder}/{name}_{N}", "zip", f"{dataset_folder}/{N}/{name}"
    )


def make_loaders(
    list_N,
    dataset_folder,
    ratio_data=1,
    batch_size=128,
    limit=1000,
    use_weighting=False,
    shuffle_train=True,
):
    """
    This function creates DataLoader objects by reading the saved train, val
    and test datasets.

    Parameters:
    ----------
    list_N: list
        List of NoC sizes for which datasets will be read and converted into
        DataLoaders. Each index should have a number `N` pertaining to a NoC
        of size `N` x `N`.

    dataset_folder: str
        Path from which the datasets will be loaded.

    ratio_data: float between 0 and 1, optional
        Only convert a portion of each dataset into DataLoader. Useful for
        loading partial datasets for quick loading and debugging. This ratio
        is applied to each individual dataset object. The final affect is that
        the returned DataLoader objects each contain only a `ratio` portion
        of their total length. Defaults to 1.

    batch_size: int, optional
        Batch size of each DataLoader. Defaults to 128.

    limit: int, optional
        Only samples with `data.y` <= limit are included in the final
        DataLoader objects. Defaults to 1000.

    use_weighting: bool, optional
        If enabled, the train DataLoader is configured to have a sampling
        frequency that is tied to each sample's wclatency occurrence in the
        dataset. Frequency buckets are of size 10. Simply put, samples
        with more common latencies will be sampled fewer times. This
        allows the extreme latencies to be "seen" more frequently by
        the NN model. Defaults to False.

    Returns
    -------
    DataLoaders pertaining to training, validation and test, in that order.
    """
    train_set = ConcatDataset(
        [
            CommonNoCDataset(dataset_folder, N, "train", limit, ratio_data)
            for N in list_N
        ]
    )
    val_set = ConcatDataset(
        [CommonNoCDataset(dataset_folder, N, "val", limit, ratio_data) for N in list_N]
    )
    test_set = ConcatDataset(
        [CommonNoCDataset(dataset_folder, N, "test", limit, ratio_data) for N in list_N]
    )

    # create DataLoader objects
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle_train
    )
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CommonNoC entry point for data generation"
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        required=True,
        help="""Sizes of the NoC for which to run the program. Space delimited
        for multiple options. Example `--N 4 8 16` for specifying sizes of 4, 8
        and 16""",
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
        "--qor_clib_path",
        type=str,
        required=True,
        help="The full path to the clib library for analysis",
    )

    parser.add_argument(
        "--traces_dir_path",
        type=str,
        required=True,
        help="The full path to the directory containing all the traces.",
    )

    parser.add_argument(
        "--train_val_test_split",
        type=float,
        nargs=3,
        required=False,
        default=[0.85, 0.05, 0.1],
        help="""A lits of length 3, specifying how much of the total dataset will
        go towards train, validation and test split. Index 0 of the list
        is for train, index 1 for validation and index 2 is for test. Must
        add upto 1. Space delimited. So for specifying a split of 0.85,
        0.05, 0.1, you should specify it as --train_val_test_split 
        `0.85 0.05 0.1`""",
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=False,
        default=f"{os.getcwd()}/datasets",
        help="Path to which the dataset will be saved.",
    )

    # parse args
    args = parser.parse_args()
    list_N = args.N
    buf_id = args.buf_id
    bp_id = args.bp_id
    port_id_mappings = {
        "north": args.north_id,
        "west": args.west_id,
        "south": args.south_id,
        "east": args.east_id,
        "pe": args.pe_id,
    }
    qor_clib_path = args.qor_clib_path
    traces_dir_path = args.traces_dir_path
    train_val_test_split = args.train_val_test_split
    dataset_folder = args.dataset_folder

    # generate datasets
    for N in list_N:
        create_dataset(
            N=N,
            buf_id=buf_id,
            bp_id=bp_id,
            port_id_mapping=port_id_mappings,
            qor_clib_path=qor_clib_path,
            traces_dir_path=traces_dir_path,
            train_val_test_split=train_val_test_split,
            dataset_folder=dataset_folder,
        )
