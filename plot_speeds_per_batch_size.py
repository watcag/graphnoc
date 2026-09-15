import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


def filter_df(df, *args, **kwargs):
    cond = None
    for key, val in kwargs.items():
        cond = cond & (df[key] == val) if cond is not None else (df[key] == val)
    return df[cond]


def get_target_df():

    df = pd.read_csv("sweep_speed.csv", sep=", ", engine="python")

    # get df for large gnn model on gpu
    df_large_gnn_gpu = filter_df(
        df, num_layers=10, emb_size=256, eval_gnn=True, model_on_gpu=True
    )
    # get df for large gnn model on cpu
    df_large_gnn_cpu = filter_df(
        df, num_layers=10, emb_size=256, eval_gnn=True, model_on_gpu=False
    )
    # get df for medium gnn model on gpu
    df_medium_gnn_gpu = filter_df(
        df, num_layers=15, emb_size=128, eval_gnn=True, model_on_gpu=True
    )
    # get df for medium gnn model on cpu
    df_medium_gnn_cpu = filter_df(
        df, num_layers=15, emb_size=128, eval_gnn=True, model_on_gpu=False
    )
    # get df for small gnn model on gpu
    df_small_gnn_gpu = filter_df(
        df, num_layers=20, emb_size=64, eval_gnn=True, model_on_gpu=True
    )
    # get df for small gnn model on cpu
    df_small_gnn_cpu = filter_df(
        df, num_layers=20, emb_size=64, eval_gnn=True, model_on_gpu=False
    )
    # get df for qor
    df_qor = filter_df(df, eval_gnn=False)

    # create empty target df
    target_df = pd.DataFrame()

    # iterate over each df type
    for df in (
        df_large_gnn_cpu,
        df_medium_gnn_cpu,
        df_small_gnn_cpu,
        df_large_gnn_gpu,
        df_medium_gnn_gpu,
        df_small_gnn_gpu,
        df_qor,
    ):
        # iterate over each N
        for noc_size in df["N"].unique():
            # iterate over batch size
            for batch_size in df["batch_size"].unique():
                # for this noc size, find maximum iterations achieved
                is_gnn = df["eval_gnn"].unique() == [True]
                is_large = (
                    is_gnn
                    and df["num_layers"].unique() == [10]
                    and df["emb_size"].unique() == [256]
                )

                is_medium = (
                    is_gnn
                    and df["num_layers"].unique() == [15]
                    and df["emb_size"].unique() == [128]
                )
                is_small = (
                    is_gnn
                    and df["num_layers"].unique() == [20]
                    and df["emb_size"].unique() == [64]
                )

                on_gpu = is_gnn and df["model_on_gpu"].unique() == [True]
                print(
                    filter_df(df, N=noc_size, batch_size=batch_size)[
                        "items_per_sec"
                    ].to_numpy()
                )
                target_df = pd.concat(
                    [
                        target_df,
                        pd.DataFrame(
                            {
                                "N": [noc_size],
                                "batch_size": [batch_size],
                                "type": (
                                    ["GNN-large"]
                                    if (is_large)
                                    else (
                                        ["GNN-medium"]
                                        if (is_medium)
                                        else (["GNN-small"] if is_small else ["QoR"])
                                    )
                                ),
                                "max_it_s": filter_df(
                                    df, N=noc_size, batch_size=batch_size
                                )["items_per_sec"].to_numpy(),
                                "hw_type": ["GPU"] if on_gpu else ["CPU"],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                print()
            # print(noc_size, filter_df(df, N=noc_size))

    return target_df


def create_save_plot(
    target_df,
    N,
    y_min,
    y_max,
    filename,
    ylabel="Iterations per Second",
    xlabel="Batch Size",
    size_x=4,
    size_y=4,
    remove_legend=True,
    remove_x_label=True,
    remove_y_label=False,
    remove_y_ticks=False,
):
    sns.set(rc={"figure.figsize": (size_x, size_y)})

    plt.figure()
    sns.set_style("ticks")

    temp_df = filter_df(target_df, N=N)

    line_plot = sns.lineplot(
        data=temp_df,
        x="batch_size",
        y="max_it_s",
        hue="type",
        style="hw_type",
        markers=True,
        markersize=10,
        palette=sns.color_palette("Dark2"),
    )
    line_plot.set_yscale("log", base=2)
    line_plot.set_xscale("log", base=2)
    line_plot.set_xticks(
        ticks=target_df["batch_size"].unique(),
        labels=target_df["batch_size"].unique(),
        rotation=60,
    )
    if remove_legend:
        line_plot.get_legend().remove()

    # set axis
    line_plot.set(ylabel=ylabel, xlabel=xlabel)
    if remove_y_label:
        line_plot.set(ylabel=None)
    if remove_x_label:
        line_plot.set(xlabel=None)

    # set y limits
    line_plot.set(ylim=(y_min, y_max))
    if remove_y_ticks:
        line_plot.set(yticklabels=[])

    fig = line_plot.get_figure()

    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    target_df = get_target_df()

    y_min, y_max = (
        target_df["max_it_s"].min() - 10000,
        target_df["max_it_s"].max() + 10000,
    )

    non_first_plot = False
    for N in target_df["N"].unique():
        create_save_plot(
            target_df=target_df,
            N=N,
            y_min=y_min,
            y_max=y_max,
            filename=f"{os.path.basename(__file__).split('.')[0]}_{N}.pdf",
            remove_legend=True,
            remove_x_label=True,
            remove_y_label=non_first_plot,
            remove_y_ticks=non_first_plot,
        )
        if not non_first_plot:
            non_first_plot = True
    # temp_df = filter_df(target_df, N=N)
    # sns.set(rc={"figure.figsize": (3, 4)})  # width=3, #height=4
    # plt.figure()
    # sns.set_style("ticks")
    # line_plot = sns.lineplot(
    #     data=temp_df, x="batch_size", y="max_it_s", hue="type", style="hw_type"
    # )
    # fig = line_plot.get_figure()
    # fig.savefig(f"{os.path.basename(__file__).split('.')[0]}_{N}.pdf")
