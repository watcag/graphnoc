import torch
import hoplite_ml_qor
import pandas as pd
from torch_geometric.data import HeteroData
import numpy as np
import math
import os
import random
import shutil
import subprocess
class HopliteGrapher:
    """
    Class to represent a HopliteBuf/BP hybrid NoC as a Heterogenous Graph
    Neural Network.
    Parameters
    ----------
    buf_id: int
        The ID for representing the core HopliteBuf switch.
    bp_id: int
        The ID for representing the core HopliteBP switch.
    port_id_mappings: dict
        The mapping of a hoplite ports to an `int` ID. Must contain 'north',
        'west', 'east', 'south' and 'pe' as keys. 
    qor_clib_path: str
        The full path to the clib library for analysis.
    """
    def __init__(self, buf_id, bp_id, port_id_mapping, qor_clib_path):
        assert all(port in port_id_mapping.keys() for port in ["north", "east", "west", "south", "pe"])
        self.qor_clib_path = qor_clib_path
        self.switch_id_mapping = {"buf": buf_id, "bp": bp_id}
        self.port_id_mapping = port_id_mapping
        self.port_indices = {"north": 0, "west": 1, "east": 2, "pe": 3, "south": 4}
        self.qor_tool = hoplite_ml_qor.pyQoR(qor_clib_path=self.qor_clib_path)

    def _get_index(self, port_name, switch_num):
        """
        For a given port name and switch number, returns the flattened
        index of the port. Note that switch counting starts from top
        left of a Hoplite NoC. See figure below
        
        
        S0 - S1 - S2 - S3
        |    |    |    |
        S4   S5   S6   S7
        |    |    |    |
        S8 - S9 - S - S
        |    |    |    |
        S    S    S    S
        
        Parameters
        ----------
        port_name: str
            The name of the port.
        switch_num: int
            Index of the switch to which the port belongs.
        
        Returns
        -------
        An int representing the flattened index of the port.
        """
        assert port_name in self.port_indices.keys()
        return (switch_num * 5) + self.port_indices[port_name]

    def _make_switch_graphs(self, N, switch_modes, data):
        """
        For a given NoC of size NxN, this makes NxN switches and represents
        each as a graph in the `data` hetero data. This particular graph 
        represents the core switch and the port as different types of graph
        nodes. Values are assigned to core switch node using `switch_id_mapping`
        and to ports using `port_id_mapping`.

        Parameters
        ----------
        N: int
            Size of the NoC where the NoC is represented as N x N
        switch_modes: list of ints
            A list of switch modes where each entry in the list is a string
            to designate the type of switch in the Hoplite NoC. Note that
            designation starts from top left and moves to right then down.
        data: HeteroData
            A heterogenous data frame.
        
        Returns
        -------
        The same heterogenous data frame with all of the NxN Hoplite switches
        represented as heterogenous graphs.
        """
        port_tensor_list = []
        for switch_num in range(N*N):
            # for each switch num, we iterate the ports (north, east, west, south)
            # in ascending order of their index relative to a switch. then, we
            # look up their node values in `port_id_mapping` and we create a list
            # of nodes
            for key in sorted(self.port_indices, key=self.port_indices.get):
                port_tensor_list.append([self.port_id_mapping[key]])
        
        # add port nodes to data frame
        data["port"].x = torch.tensor(port_tensor_list, dtype=torch.float)


        # add switch nodes
        # TODO: i am not multiplying the self.switch_id_mapping[mode] with N.
        # typing this here in case the model does not train well. it was 
        # self.switch_id_mapping[mode]*N before
        data["switch"].x = torch.tensor([self.switch_id_mapping[mode] for mode in switch_modes], dtype=torch.float)[
            :, None
        ]

        # captures which port can route to which port
        port_port_list = []

        # bidirectional connections between a core switch node and its ports.
        port_switch_list = []  # connects central switch node to its port nodes
        switch_port_list = []  # connect port nodes to its central switch node. 

        # we generate separate graphs for each of the N x N switches in the NoC
        for switch_count in range(N * N):
            # get index of ports
            north = self._get_index("north", switch_count)
            west = self._get_index("west", switch_count)
            east = self._get_index("east", switch_count)
            pe = self._get_index("pe", switch_count)
            south = self._get_index("south", switch_count)

            # capture relationship of "which port can route to which port"
            port_port_list.append(torch.tensor(([north], [east])))  # north to east
            port_port_list.append(torch.tensor(([north], [south])))  # north to south
            port_port_list.append(torch.tensor(([north], [pe])))  # north to pe

            port_port_list.append(torch.tensor(([west], [east])))  # west to east
            port_port_list.append(torch.tensor(([west], [south])))  # west to south
            port_port_list.append(torch.tensor(([west], [pe])))  # west to pe

            port_port_list.append(torch.tensor(([pe], [south])))  # pe to south
            port_port_list.append(torch.tensor(([pe], [east])))  # pe to east

            # connect core switches and their ports as bidirectional edges
            port_switch_list.append(torch.tensor(([north], [switch_count])))  # north and switch
            switch_port_list.append(torch.tensor(([switch_count], [north])))  # switch and north

            port_switch_list.append(torch.tensor(([east], [switch_count])))  # east and switch
            switch_port_list.append(torch.tensor(([switch_count], [east])))  # switch and east

            port_switch_list.append(torch.tensor(([south], [switch_count])))  # south and switch
            switch_port_list.append(torch.tensor(([switch_count], [south])))  # switch and south

            port_switch_list.append(torch.tensor(([west], [switch_count])))  # west and switch
            switch_port_list.append(torch.tensor(([switch_count], [west])))  # switch and west

            port_switch_list.append(torch.tensor(([pe], [switch_count])))  # pe and switch
            switch_port_list.append(torch.tensor(([switch_count], [pe])))  # switch and pe

        # making the actual connections in COO format. See 
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        # for details on COO format.
        # This is a heterogenous connection. See
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        # for details on heterogenous graphs.
        data["port", "routes", "port"].edge_index = torch.cat(port_port_list, axis=1)
        data["port", "switch_con", "switch"].edge_index = torch.cat(
            port_switch_list, axis=1
        )
        data["switch", "switch_con", "port"].edge_index = torch.cat(
            switch_port_list, axis=1
        )

        return data

    def _make_noc_topology(self, N, data):
        """
        For a given NoC of size NxN, this connects the switches together
        and represents the NoC as a graph in the `data` hetero data.

        Parameters
        ----------
        N: int
            Size of the NoC where the NoC is represented as N x N
        data: HeteroData
            A heterogenous data frame. This data frame must already have
            individual switches represented as graphs. 

        Returns
        -------
        The same heterogenous data frame with all of the NxN switch graphs
        connected together to reflect Hoplite's topology.
        """
        # list to capture edges in COO format. which port of one switch
        # connects to which port of the other switch. basically connections
        # between switches.
        port_port_list = []

        # make horizontal connections first
        for row in range(N):
            for col in range(N):
                cur_switch = row * N + col

                # cur switch connects to either switch on its right
                # or first switch in its row if wrapping around
                next_switch = (row*N) if col == N-1 else (cur_switch+1)

                # connect current switch's east to next switch's west
                port_port_list.append(
                    torch.tensor(
                        (
                            [self._get_index("east", cur_switch)],
                            [self._get_index("west", next_switch)],
                        )
                    )
                )
                # TODO: commenting this out below for now. for the
                # successfully trained model, i had this uncommented
                # port_port_list.append(
                #     torch.tensor(
                #         (
                #             [self._get_index("west", next_switch)],
                #             [self._get_index("east", cur_switch)],
                #         )
                #     )
                # )

        # now make vertical connections
        for col in range(N):
            for row in range(N):
                cur_switch = row * N + col

                # cur switch connects to either switch below
                # or first switch in its column if wrapping around
                next_switch = col if row == N-1 else (cur_switch + N)

                # connect south to next switch's north
                port_port_list.append(
                    torch.tensor(
                        (
                            [self._get_index("south", cur_switch)],
                            [self._get_index("north", next_switch)],
                        )
                    )
                )
                # TODO: commenting this out below for now. for the
                # successfully trained model, i had this uncommented
                # port_port_list.append(
                #     torch.tensor(
                #         (
                #             [self._get_index("north", next_switch)],
                #             [self._get_index("south", cur_switch)],
                #         )
                #     )
                # )

        # making the actual connections in COO format. See 
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        # for details on COO format.
        # This is a heterogenous connection. See
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        # for details on heterogenous graphs.
        data["port", "noc_con", "port"].edge_index = torch.cat(port_port_list, axis=1)

        return data

    # TODO: rate_user must be either an int or a list. there can not be anything else. we are not reading
    # from trace file. same for burst, which should be an int
    def _make_trace_graphs_hoplite_ml(
        self, N, switch_modes, data, trace_file, rate_user, burst_user
    ):
        """
        Represents the particular application being routed on the NxN
        NoC as a graph.

        Parameters
        ----------
        N: int
            Size of the NoC where the NoC is represented as N x N
        
        switch_modes: list of ints
            A list of switch modes where each entry in the list is a string
            to designate the type of switch in the Hoplite NoC. Note that
            designation starts from top left and moves to right then down.
        data: HeteroData
            A heterogenous data frame. This data frame must already have
            individual switches represented as graphs.
        trace_file: str
            Absolute path to the application's trace file.
        rate_user: float
            Rate of each trace in the application. Single rate is applied to
            all traces. Must be a float between 0 and 1
        burst_user: int
            Burst size of each trace in the application. Same burst size is
            applied to all traces. 
        """
        # read traffic trace file
        df = pd.read_csv(trace_file, sep=r",\s+", engine="python")

        # each trace in the application is represented as node between the source
        # PE and destination PE. add these nodes below.
        # we scale the rate nodes' values to lie between 0 and 100. this might not
        # be actually needed but for the current training recipe, we found convergence
        # to be quicker with scaling.
        data["trace"].x = torch.ones(len(df), 1, dtype=torch.float) * rate_user * 100

        # calculate wclatency and network stability
        self.qor_tool.analyze_network(x=switch_modes, N=N, df=df, burst_user=burst_user, rate_user=rate_user)
        wclatency = self.qor_tool.wclatency
        network_unstable = self.qor_tool.network_unstable

        data.y = torch.tensor(wclatency, dtype=torch.float)
        data.network_unstable = network_unstable

        port_traffic_trace = []
        trace_traffic_port = []

        # each trace in the application is represented as node between the source
        # PE and destination PE
        for index, row in df.iterrows():
            # convert (X,Y) designation of src, dst in trace file to the
            # flattened numbering we use.
            src, dst = int(row["sY"] * N + row["sX"]), int(row["dY"] * N + row["dX"])

            # make edge from PE port of src to the trace's node
            port_traffic_trace.append(
                torch.tensor(([self._get_index("pe", src)], [index]))
            )

            # make edge from trace's node to PE port of destination
            trace_traffic_port.append(
                torch.tensor(([index], [self._get_index("pe", dst)]))
            )

        # making the actual connections in COO format. See 
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        # for details on COO format.
        # This is a heterogenous connection. See
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        # for details on heterogenous graphs.
        data["port", "traffic", "trace"].edge_index = torch.cat(
            port_traffic_trace, axis=1
        )
        data["trace", "traffic", "port"].edge_index = torch.cat(
            trace_traffic_port, axis=1
        )

        return data

    def _validate_inputs(self, N, switch_modes, trace_file, rate_user, burst_user):
        assert burst_user == 1, "other burst values not supported"
        assert np.all([mode in ["bp", "buf"] for mode in switch_modes])
        assert len(switch_modes) == N*N
        assert type(rate_user) == float
        assert 0 < rate_user <=1

    def make_hl_ml_gnn(self, N, switch_modes, trace_file, rate_user, burst_user):
        # empty hetero data frame.
        data = HeteroData()

        # connect switch nodes and port nodes together to represent all the
        # switches in the NoC as a graph
        data = self._make_switch_graphs(N=N, switch_modes=switch_modes, data=data)

        # connect together switch graphs to make NoC graph
        data = self._make_noc_topology(N, data)

        # represent the application being routed as a graph
        data = self._make_trace_graphs_hoplite_ml(
            N=N,
            switch_modes=np.array(
                [0 if mode == "buf" else 1 for mode in switch_modes], dtype=np.double
            ),
            data=data,
            trace_file=trace_file,
            rate_user=rate_user,
            burst_user=burst_user,
        )
        return data
    

class BFT0Grapher:
    def __init__(self, t_id, port_id_mapping, bft_path):
        assert (
            "dl" in port_id_mapping.keys()
            and "dr" in port_id_mapping.keys()
            and "u" in port_id_mapping.keys()
            and len(port_id_mapping.keys()) == 3
        )
        self.mappings = {"t": t_id}
        self.port_id_mapping = port_id_mapping
        self.port_indices = {"dl": 0, "dr": 1, "u": 2}
        self.bft_path=bft_path

    def _get_index(self, port_name, switch_num):
        """
        For a given port name and switch number, returns the flattened
        index of the port
        """
        assert port_name in self.port_indices.keys()
        return (switch_num * 3) + self.port_indices[port_name]

    def _make_switch_graphs(self, num_switches, data):
        # make edges. connect ports to different ports
        port_port_list = []
        port_switch_list = []
        switch_port_list = []
        for switch_count in range(num_switches):
            dl = self._get_index("dl", switch_count)
            dr = self._get_index("dr", switch_count)
            u = self._get_index("u", switch_count)

            # connect dl to u
            port_port_list.append(torch.tensor(([dl], [u])))
            # connect dl to dr
            port_port_list.append(torch.tensor(([dl], [dr])))

            # connect dr to u
            port_port_list.append(torch.tensor(([dr], [u])))
            # connect dr to dl
            port_port_list.append(torch.tensor(([dr], [dl])))

            # connect u to dl
            port_port_list.append(torch.tensor(([u], [dl])))
            # connect u to dr
            port_port_list.append(torch.tensor(([u], [dr])))

            # connect dl and switch
            port_switch_list.append(torch.tensor(([dl], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [dl])))

            # connect dr and switch
            port_switch_list.append(torch.tensor(([dr], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [dr])))

            # connect u and switch
            port_switch_list.append(torch.tensor(([u], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [u])))

        data["port", "routes", "port"].edge_index = torch.cat(port_port_list, axis=1)
        data["port", "switch_con", "switch"].edge_index = torch.cat(
            port_switch_list, axis=1
        )
        data["switch", "switch_con", "port"].edge_index = torch.cat(
            switch_port_list, axis=1
        )

        return data

    # Remember that the NoC switch count starts from top in case
    # of BFT0. see example below
    #           X-0
    #          / \
    #         /   \
    #        /     \
    #      1-X       X-2
    #       /\       /\
    #      /  \     /  \
    #     /    \   /    \
    #   3-X    X-4 5-X  X-6

    # is switch 0. Then L0, right of left most is switch 1 and so on.
    def _make_noc_topology(self, num_switches, data):
        port_port_list = []
        # we start from switch 0 to num_switches
        # for each switch, we connect its dl and dr to its
        # corresponding u switches

        # another import thing to note is the naming convention
        #         cur_switch
        #             *
        #
        #      *             *
        # left_switch    right_switch
        for switch_index in range(num_switches // 2):
            cur_switch = switch_index
            left_switch = (cur_switch * 2) + 1
            right_switch = left_switch + 1

            dl_cur_switch = self._get_index("dl", cur_switch)
            dr_cur_switch = self._get_index("dr", cur_switch)

            u_left_switch = self._get_index("u", left_switch)
            u_right_switch = self._get_index("u", right_switch)

            # connect dl of cur_switch to u of left_switch
            port_port_list.append(torch.tensor(([dl_cur_switch], [u_left_switch])))
            port_port_list.append(torch.tensor(([u_left_switch], [dl_cur_switch])))

            # connect dr of cur_switch to u of right_switch
            port_port_list.append(torch.tensor(([dr_cur_switch], [u_right_switch])))
            port_port_list.append(torch.tensor(([u_right_switch], [dr_cur_switch])))

        data["port", "noc_con", "port"].edge_index = torch.cat(port_port_list, axis=1)

        return data

    # from trace file. same for burst, which should be an int
    def _make_trace_graphs(self, N, data, trace, rate_user, id, num_packets=1024):
        cwd = os.getcwd()

        veri_dir = f"{cwd}/{id}"

        os.makedirs(veri_dir, exist_ok=True)

        assert type(rate_user) in [float, int]

        list_trace_x = []
        port_traffic_trace = []
        trace_traffic_port = []
        for src in range(N):
            dst_set = set()
            if trace == "random":
                # choose a random
                dst_set.add(random.choice([x for x in range(N) if x != src]))
            elif trace == "local":
                dst_set.add(
                    random.choice(
                        [
                            x
                            for x in range(N)
                            if (x != src and abs(x - src) <= math.sqrt(N))
                        ]
                    )
                )
            else:
                df = pd.read_csv(
                    f"{self.bft_path}/bench/{trace}/{N}/autogen_{src}.trace",
                    sep=r",\s+",
                    engine="python",
                )

                # identify unique destinations
                for dst in df.to_numpy():
                    val_string = dst[0]
                    if val_string[0] != "f":
                        dst_set.add(int(val_string[1:], 16))
                dst_set.remove(src)

            # create updated trace file
            cur_length = 0
            with open(f"{veri_dir}/autogen_{src}.trace", "w") as fp:
                while True:
                    if len(dst_set) == 0:
                        break
                    if cur_length == num_packets:
                        break
                    for dst in dst_set:
                        hex_string = "0" + f"{dst:02x}"
                        fp.write("%s\n" % hex_string)
                        cur_length += 1
                        if cur_length == num_packets:
                            break
                fp.write("f00\n")

            # create node from src to each unique destination
            src_index = self._get_index(
                "dl" if src % 2 == 0 else "dr", (src // 2) + (N // 2) - 1
            )
            for dst in dst_set:
                # create trace node and add it to list_trace_x
                list_trace_x.append(torch.ones(1, 1) * rate_user)

                trace_node_index = len(list_trace_x) - 1

                # make connection from src to trace node
                port_traffic_trace.append(
                    torch.tensor(([src_index], [trace_node_index]))
                )

                # make connection from trace node to dst
                dst_index = self._get_index(
                    "dl" if dst % 2 == 0 else "dr", (dst // 2) + (N // 2) - 1
                )
                trace_traffic_port.append(
                    torch.tensor(([trace_node_index], [dst_index]))
                )

        # create edges
        data["port", "traffic", "trace"].edge_index = torch.cat(
            port_traffic_trace, axis=1
        )
        data["trace", "traffic", "port"].edge_index = torch.cat(
            trace_traffic_port, axis=1
        )
        data["trace"].x = torch.cat(list_trace_x, axis=0)

        data.y = self.calc_wclatency(N, rate_user, veri_dir, cwd)
        shutil.rmtree(veri_dir)

        return data

    def calc_wclatency(self, N, rate, veri_dir, cwd):
        os.chdir(veri_dir)

        for dir_name in ["rtl", "includes", "tb"]:
            src_files = os.listdir(f"{self.bft_path}/{dir_name}")
            for src_file in src_files:
                full_file_name = f"{self.bft_path}/{dir_name}/{src_file}"
                shutil.copy(full_file_name, veri_dir)

        os.system(
            f"verilator -Wno-WIDTH -Wno-COMBDLY -Od -Wno-WIDTHCONCAT -DTREE -DREAL -DSIM -GWRAP=1 -GSIGMA=4 -GN={N} -GRATE={rate} -GPAT=0 --top-module bft --cc bft.v client_bp.v client_bp_top.v mux.v pi_switch_top.v pi_switch.v t_switch.v t_route.v pi_route.v t_switch_top.v bp.v --exe bft_tb.c >/dev/null 2>&1"
        )
        os.system("make -C obj_dir -j -f Vbft.mk Vbft >/dev/null 2>&1")
        os.system("./obj_dir/Vbft > test.log")

        os.system(
            'cat test.log | grep "Attempted" | cut -d" " -f1,3 | sed "s/Time//" | sed "s/://" | sed "s/packetid=//" | sort -t" " -k2 -V | sed "s/ /,/" > attempt.log'
        )
        os.system(
            'cat test.log | grep "Sent" | cut -d" " -f1,9 | sed "s/Time//" | sed "s/://" | sed "s/\(.*\),/\1/" | sed "s/packetid=//" | sort -t" " -k2 -V | sed "s/ /,/" > sentq.log'
        )
        os.system(
            'cat test.log | grep "Received" | cut -d" " -f1,7 | sed "s/Time//" | sed "s/://" | sed "s/\(.*\),/\1/" | sed "s/data=//" | sort -t" " -k2 -V | sed "s/ /,/" > recvq.log'
        )
        os.system('paste -d"," attempt.log sentq.log recvq.log > latency.log')
        os.system(
            """awk -F"," '{print $5-$1","$5-$3","$3-$1}' latency.log > latency-fixed.log"""
        )
        wc_latency_output = float(
            subprocess.run(
                ['cat latency-fixed.log | cut -d"," -f 2 | sort -n | tail -1'],
                capture_output=True,
                shell=True,
            ).stdout
        )

        os.chdir(cwd)
        return wc_latency_output

    def make_gnn(self, N, trace, rate_user, id):
        assert math.log2(N) % 1 == 0, f"{N} must be power of 2"

        data = HeteroData()
        num_switches = N - 1

        # add switch nodes
        data["switch"].x = torch.tensor(
            [self.mappings["t"] * N] * num_switches, dtype=torch.float
        )[:, None]

        # add port nodes
        port_tensor_list = []
        for switch_id in range(num_switches):
            for key in sorted(self.port_indices, key=self.port_indices.get):
                port_tensor_list.append([self.port_id_mapping[key]])
        data["port"].x = torch.tensor(port_tensor_list, dtype=torch.float)

        # connect switch nodes and port nodes together to make a switch graph
        data = self._make_switch_graphs(num_switches, data)

        # connect together switch graphs to make noc graph
        data = self._make_noc_topology(num_switches, data)

        # add trace connections
        data = self._make_trace_graphs(N, data, trace, rate_user, id)

        return data

class BFT3Grapher:
    def __init__(self, pi_id, t_id, port_id_mapping,bft_path):
        assert (
            "dl" in port_id_mapping.keys()
            and "dr" in port_id_mapping.keys()
            and "ul" in port_id_mapping.keys()
            and "ur" in port_id_mapping.keys()
            and len(port_id_mapping.keys()) == 4
        )
        self.mappings = {"pi": pi_id, "t": t_id}
        self.port_id_mapping = port_id_mapping
        self.port_indices = {"dl": 0, "dr": 1, "ul": 2, "ur": 3}
        self.bft_path=bft_path

    def _get_index(self, port_name, switch_num):
        """
        For a given port name and switch number, returns the flattened
        index of the port
        """
        assert port_name in self.port_indices.keys()
        return (switch_num * 4) + self.port_indices[port_name]

    def _make_switch_graphs(self, num_switches, data):
        # make edges. connect ports to different ports
        port_port_list = []
        port_switch_list = []
        switch_port_list = []
        for switch_count in range(num_switches):
            dl = self._get_index("dl", switch_count)
            dr = self._get_index("dr", switch_count)
            ul = self._get_index("ul", switch_count)
            ur = self._get_index("ur", switch_count)

            # connect dl to ul
            port_port_list.append(torch.tensor(([dl], [ul])))
            # connect dl to dr
            port_port_list.append(torch.tensor(([dl], [dr])))

            # connect dr to ur
            port_port_list.append(torch.tensor(([dr], [ur])))
            # connect dr to dl
            port_port_list.append(torch.tensor(([dr], [dl])))

            # connect ul to dl
            port_port_list.append(torch.tensor(([ul], [dl])))
            # connect ul to dr
            port_port_list.append(torch.tensor(([ul], [dr])))

            # connect ur to dl
            port_port_list.append(torch.tensor(([ur], [dl])))
            # connect ur to dr
            port_port_list.append(torch.tensor(([ur], [dr])))

            # connect dl and switch
            port_switch_list.append(torch.tensor(([dl], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [dl])))

            # connect dr and switch
            port_switch_list.append(torch.tensor(([dr], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [dr])))

            # connect ul and switch
            port_switch_list.append(torch.tensor(([ul], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [ul])))

            # connect ur and switch
            port_switch_list.append(torch.tensor(([ur], [switch_count])))
            switch_port_list.append(torch.tensor(([switch_count], [ur])))

        data["port", "routes", "port"].edge_index = torch.cat(port_port_list, axis=1)
        data["port", "switch_con", "switch"].edge_index = torch.cat(
            port_switch_list, axis=1
        )
        data["switch", "switch_con", "port"].edge_index = torch.cat(
            switch_port_list, axis=1
        )

        return data

    # Remember that the NoC switch count starts from bottom left. So L0 leftmost
    # is switch 0. Then L0, right of left most is switch 1 and so on.
    def _make_noc_topology(self, num_switches, num_levels, data):
        num_switches_per_row = num_switches / num_levels
        assert num_switches_per_row % 1 == 0
        num_switches_per_row = int(num_switches_per_row)
        # define a dict that maps switch index to wether its uppers have
        # been defined
        uppers_connected = {}
        for cur_switch_index in range(num_switches):
            uppers_connected[cur_switch_index] = False

        port_port_list = []

        # we start from level 0 to num_levels and then iterate from
        # left most to right most switch and then connect their uppers

        # another import thing to note is the naming convention

        #  up_switch    diag_switch
        #      *            *
        #
        #      *            *
        # cur_switch    right_switch
        for row in range(num_levels - 1):
            for col in range(num_switches_per_row):
                cur_switch = (row * num_switches_per_row) + col

                # if the uppers of this switch have been defined, we can ignore it.
                if not uppers_connected[cur_switch]:
                    # calculate indices of other switches in the xbar
                    up_switch = cur_switch + num_switches_per_row
                    right_switch = (row * num_switches_per_row) + col + (2**row)
                    diag_switch = right_switch + num_switches_per_row

                    # connect ul of cur_switch to dl of up_switch and vice versa
                    ul_cur_switch = self._get_index("ul", cur_switch)
                    dl_up_switch = self._get_index("dl", up_switch)

                    port_port_list.append(
                        torch.tensor(([ul_cur_switch], [dl_up_switch]))
                    )
                    port_port_list.append(
                        torch.tensor(([dl_up_switch], [ul_cur_switch]))
                    )

                    # connect ur of cur switch to dl of diag_switch
                    ur_cur_switch = self._get_index("ur", cur_switch)
                    dl_diag_switch = self._get_index("dl", diag_switch)

                    port_port_list.append(
                        torch.tensor(([ur_cur_switch], [dl_diag_switch]))
                    )
                    port_port_list.append(
                        torch.tensor(([dl_diag_switch], [ur_cur_switch]))
                    )

                    uppers_connected[cur_switch] = True

                    # connect ul of right_switch to dr of up_switch and vice versa
                    ul_right_switch = self._get_index("ul", right_switch)
                    dr_up_switch = self._get_index("dr", up_switch)

                    port_port_list.append(
                        torch.tensor(([ul_right_switch], [dr_up_switch]))
                    )
                    port_port_list.append(
                        torch.tensor(([dr_up_switch], [ul_right_switch]))
                    )

                    # connect ur of right switch to dr of diag switch and vice versa
                    ur_right_switch = self._get_index("ur", right_switch)
                    dr_diag_switch = self._get_index("dr", diag_switch)

                    port_port_list.append(
                        torch.tensor(([ur_right_switch], [dr_diag_switch]))
                    )
                    port_port_list.append(
                        torch.tensor(([dr_diag_switch], [ur_right_switch]))
                    )

                    uppers_connected[right_switch] = True
        data["port", "noc_con", "port"].edge_index = torch.cat(port_port_list, axis=1)

        return data

    # from trace file. same for burst, which should be an int
    def _make_trace_graphs(self, N, data, trace, rate_user, id, num_packets=1024):
        cwd = os.getcwd()

        veri_dir = f"{cwd}/{id}"

        os.makedirs(veri_dir, exist_ok=True)

        assert type(rate_user) in [float, int]

        list_trace_x = []
        port_traffic_trace = []
        trace_traffic_port = []
        for src in range(N):
            dst_set = set()
            if trace == "random":
                # choose a random
                dst_set.add(random.choice([x for x in range(N) if x != src]))
            elif trace == "local":
                dst_set.add(
                    random.choice(
                        [
                            x
                            for x in range(N)
                            if (x != src and abs(x - src) <= math.sqrt(N))
                        ]
                    )
                )
            else:
                df = pd.read_csv(
                    f"{self.bft_path}/bench/{trace}/{N}/autogen_{src}.trace",
                    sep=r",\s+",
                    engine="python",
                )

                # identify unique destinations
                for dst in df.to_numpy():
                    val_string = dst[0]
                    if val_string[0] != "f":
                        dst_set.add(int(val_string[1:], 16))
                dst_set.remove(src)

            # create updated trace file
            cur_length = 0
            with open(f"{veri_dir}/autogen_{src}.trace", "w") as fp:
                while True:
                    if len(dst_set) == 0:
                        break
                    if cur_length == num_packets:
                        break
                    for dst in dst_set:
                        hex_string = "0" + f"{dst:02x}"
                        fp.write("%s\n" % hex_string)
                        cur_length += 1
                        if cur_length == num_packets:
                            break
                fp.write("f00\n")

            # create node from src to each unique destination
            src_index = self._get_index("dl" if src % 2 == 0 else "dr", src // 2)
            for dst in dst_set:
                # create trace node and add it to list_trace_x
                # TODO: we multiply divide etc in hoplite. do same numerical precision here as well but make
                # sure input to bft sim is between 0 and 100.
                list_trace_x.append(torch.ones(1, 1) * rate_user)

                trace_node_index = len(list_trace_x) - 1

                # make connection from src to trace node
                port_traffic_trace.append(
                    torch.tensor(([src_index], [trace_node_index]))
                )

                # make connection from trace node to dst
                dst_index = self._get_index("dl" if dst % 2 == 0 else "dr", dst // 2)
                trace_traffic_port.append(
                    torch.tensor(([trace_node_index], [dst_index]))
                )

        # create edges
        data["port", "traffic", "trace"].edge_index = torch.cat(
            port_traffic_trace, axis=1
        )
        data["trace", "traffic", "port"].edge_index = torch.cat(
            trace_traffic_port, axis=1
        )
        data["trace"].x = torch.cat(list_trace_x, axis=0)

        data.y = self.calc_wclatency(N, rate_user, veri_dir, cwd)
        shutil.rmtree(veri_dir)

        return data

    def calc_wclatency(self, N, rate, veri_dir, cwd):
        os.chdir(veri_dir)

        for dir_name in ["rtl", "includes", "tb"]:
            src_files = os.listdir(f"{self.bft_path}/{dir_name}")
            for src_file in src_files:
                full_file_name = f"{self.bft_path}/{dir_name}/{src_file}"
                shutil.copy(full_file_name, veri_dir)

        os.system(
            f"verilator -Wno-WIDTH -Wno-COMBDLY -Od -Wno-WIDTHCONCAT -DXBAR -DREAL -DSIM -GWRAP=1 -GSIGMA=4 -GN={N} -GRATE={rate} -GPAT=0 --top-module bft --cc bft.v client_bp.v client_bp_top.v mux.v pi_switch_top.v pi_switch.v t_switch.v t_route.v pi_route.v t_switch_top.v bp.v --exe bft_tb.c >/dev/null 2>&1"
        )
        os.system("make -C obj_dir -j -f Vbft.mk Vbft >/dev/null 2>&1")
        os.system("./obj_dir/Vbft > test.log")

        os.system(
            'cat test.log | grep "Attempted" | cut -d" " -f1,3 | sed "s/Time//" | sed "s/://" | sed "s/packetid=//" | sort -t" " -k2 -V | sed "s/ /,/" > attempt.log'
        )
        os.system(
            'cat test.log | grep "Sent" | cut -d" " -f1,9 | sed "s/Time//" | sed "s/://" | sed "s/\(.*\),/\1/" | sed "s/packetid=//" | sort -t" " -k2 -V | sed "s/ /,/" > sentq.log'
        )
        os.system(
            'cat test.log | grep "Received" | cut -d" " -f1,7 | sed "s/Time//" | sed "s/://" | sed "s/\(.*\),/\1/" | sed "s/data=//" | sort -t" " -k2 -V | sed "s/ /,/" > recvq.log'
        )
        os.system('paste -d"," attempt.log sentq.log recvq.log > latency.log')
        os.system(
            """awk -F"," '{print $5-$1","$5-$3","$3-$1}' latency.log > latency-fixed.log"""
        )
        wc_latency_output = float(
            subprocess.run(
                ['cat latency-fixed.log | cut -d"," -f 2 | sort -n | tail -1'],
                capture_output=True,
                shell=True,
            ).stdout
        )

        os.chdir(cwd)
        return wc_latency_output

    # TODO: we should probably also add the cross routing vs direct routing
    # thingy we did  in trets. for now, we are ignorning that and using the default
    # pi of fccm. we can add this later if needed.
    def make_gnn(self, N, trace, rate_user, id):
        data = HeteroData()

        # calculate number of levels of bft
        num_levels = math.log2(N)
        assert num_levels % 1 == 0, f"{N} must be power of 2"
        num_levels = int(num_levels)

        num_switches = int(num_levels * N / 2)

        # add switch nodes
        data["switch"].x = torch.tensor(
            [self.mappings["pi"] * N] * num_switches, dtype=torch.float
        )[:, None]

        # add port nodes
        port_tensor_list = []
        for switch_id in range(num_switches):
            for key in sorted(self.port_indices, key=self.port_indices.get):
                port_tensor_list.append([self.port_id_mapping[key]])
        data["port"].x = torch.tensor(port_tensor_list, dtype=torch.float)

        # connect switch nodes and port nodes together to make a switch graph
        data = self._make_switch_graphs(num_switches, data)

        # connect together switch graphs to make noc graph
        data = self._make_noc_topology(num_switches, num_levels, data)

        # add trace connections
        data = self._make_trace_graphs(N, data, trace, rate_user, id)

        return data
