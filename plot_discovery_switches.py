import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


def filter_df(df, *args, **kwargs):
    cond = None
    for key, val in kwargs.items():
        cond = cond & (df[key] == val) if cond is not None else (df[key] == val)
    return df[cond]


def get_individual_df(N):
    full_df = pd.read_csv("sweep_discover_switches.csv", sep=", ", engine="python")
    df = filter_df(full_df, N=N)

    # get df for large gnn model on gpu
    df_large_gnn_gpu = filter_df(
        df,
        num_layers=10,
        emb_size=256,
        model_type="graph_sage_class_reg",
        device="cuda",
    )
    df_large_gnn_gpu["model_type"] = "GNN_Large_GPU"

    # get df for large gnn model hybrid on gpu
    df_large_gnn_hybrid_gpu = filter_df(
        df,
        num_layers=10,
        emb_size=256,
        model_type="graph_sage_class_reg_hybrid",
        device="cuda",
    )
    df_large_gnn_hybrid_gpu["model_type"] = "GNN_Hybrid_Large_GPU"

    # get df for large gnn model on cpu
    df_large_gnn_cpu = filter_df(
        df, num_layers=10, emb_size=256, model_type="graph_sage_class_reg", device="cpu"
    )
    df_large_gnn_cpu["model_type"] = "GNN_Large_CPU"

    # get df for large gnn model hybrid on cpu
    df_large_gnn_hybrid_cpu = filter_df(
        df,
        num_layers=10,
        emb_size=256,
        model_type="graph_sage_class_reg_hybrid",
        device="cpu",
    )
    df_large_gnn_hybrid_cpu["model_type"] = "GNN_Hybrid_Large_CPU"

    # get df for medium gnn model on gpu
    df_medium_gnn_gpu = filter_df(
        df,
        num_layers=15,
        emb_size=128,
        model_type="graph_sage_class_reg",
        device="cuda",
    )
    df_medium_gnn_gpu["model_type"] = "GNN_Medium_GPU"

    # get df for medium gnn model hybrid on gpu
    df_medium_gnn_hybrid_gpu = filter_df(
        df,
        num_layers=15,
        emb_size=128,
        model_type="graph_sage_class_reg_hybrid",
        device="cuda",
    )
    df_medium_gnn_hybrid_gpu["model_type"] = "GNN_Hybrid_Medium_GPU"

    # get df for medium gnn model on cpu
    df_medium_gnn_cpu = filter_df(
        df, num_layers=15, emb_size=128, model_type="graph_sage_class_reg", device="cpu"
    )
    df_medium_gnn_cpu["model_type"] = "GNN_Medium_CPU"

    # get df for medium gnn model hybrid on cpu
    df_medium_gnn_hybrid_cpu = filter_df(
        df,
        num_layers=15,
        emb_size=128,
        model_type="graph_sage_class_reg_hybrid",
        device="cpu",
    )
    df_medium_gnn_hybrid_cpu["model_type"] = "GNN_Hybrid_Medium_CPU"

    # get df for small gnn model on gpu
    df_small_gnn_gpu = filter_df(
        df, num_layers=20, emb_size=64, model_type="graph_sage_class_reg", device="cuda"
    )
    df_small_gnn_gpu["model_type"] = "GNN_Small_GPU"

    # get df for small gnn model hybrid on gpu
    df_small_gnn_hybrid_gpu = filter_df(
        df,
        num_layers=20,
        emb_size=64,
        model_type="graph_sage_class_reg_hybrid",
        device="cuda",
    )
    df_small_gnn_hybrid_gpu["model_type"] = "GNN_Hybrid_Small_GPU"

    # get df for small gnn model on cpu
    df_small_gnn_cpu = filter_df(
        df, num_layers=20, emb_size=64, model_type="graph_sage_class_reg", device="cpu"
    )
    df_small_gnn_cpu["model_type"] = "GNN_Small_CPU"

    # get df for small gnn model hybrid on cpu
    df_small_gnn_hybrid_cpu = filter_df(
        df,
        num_layers=20,
        emb_size=64,
        model_type="graph_sage_class_reg_hybrid",
        device="cpu",
    )
    df_small_gnn_hybrid_cpu["model_type"] = "GNN_Hybrid_Small_CPU"

    # get df for small gnn model hybrid on cpu
    df_qor_cpu = filter_df(df, model_type="qor", device="cpu")
    df_qor_cpu["model_type"] = "Hoplite_Tool"

    target_df_large = pd.concat(
        [
            df_large_gnn_cpu,
            df_large_gnn_hybrid_cpu,
            df_large_gnn_gpu,
            df_large_gnn_hybrid_gpu,
            df_qor_cpu,
        ],
        ignore_index=False,
    )

    target_df_medium = pd.concat(
        [
            df_medium_gnn_cpu,
            df_medium_gnn_hybrid_cpu,
            df_medium_gnn_gpu,
            df_medium_gnn_hybrid_gpu,
            df_qor_cpu,
        ],
        ignore_index=False,
    )

    target_df_small = pd.concat(
        [
            df_small_gnn_cpu,
            df_small_gnn_hybrid_cpu,
            df_small_gnn_gpu,
            df_small_gnn_hybrid_gpu,
            df_qor_cpu,
        ],
        ignore_index=False,
    )

    return target_df_large, target_df_medium, target_df_small


def create_and_save_plot(
    target_df,
    y_min,
    y_max,
    filename,
    ylabel="Min. Worst Case Latency (cycles)",
    xlabel="Total Time (s)",
    size_x=3,
    size_y=4,
    remove_legend=True,
    remove_x_label=True,
    remove_y_label=False,
    remove_y_ticks=False,
):
    
    # set style
    sns.set(rc={"figure.figsize": (size_x, size_y)})  # width=3, #height=4

    plt.figure()
    sns.set_style("ticks")
    scatter_plot = sns.scatterplot(
        data=target_df,
        x="total_time",
        y="min_wclatency",
        hue="rate",
        style="model_type",
        s=150,
        palette=sns.color_palette("Dark2")
    )
    if remove_legend:
        scatter_plot.get_legend().remove()

    # set axis
    scatter_plot.set(ylabel=ylabel, xlabel=xlabel)
    if remove_y_label:
        scatter_plot.set(ylabel=None)
    if remove_x_label:
        scatter_plot.set(xlabel=None)

    # set y limits
    scatter_plot.set(ylim=(y_min, y_max))
    if remove_y_ticks:
        scatter_plot.set(yticklabels=[])

    # save pdf
    fig = scatter_plot.get_figure()
    fig.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    N = 4

    target_df_large, target_df_medium, target_df_small = get_individual_df(N)
    overall_df = pd.concat(
        [target_df_small, target_df_medium, target_df_large], ignore_index=False
    )

    y_min, y_max = overall_df["min_wclatency"].min()-10, overall_df["min_wclatency"].max()+10

    create_and_save_plot(
        target_df=target_df_small,
        y_min=y_min,
        y_max=y_max,
        filename=f"{os.path.basename(__file__).split('.')[0]}_{N}_small.pdf",
    )

    create_and_save_plot(
        target_df=target_df_medium,
        y_min=y_min,
        y_max=y_max,
        filename=f"{os.path.basename(__file__).split('.')[0]}_{N}_medium.pdf",
        remove_y_label=True,
        remove_y_ticks=True,
    )

    create_and_save_plot(
        target_df=target_df_large,
        y_min=y_min,
        y_max=y_max,
        filename=f"{os.path.basename(__file__).split('.')[0]}_{N}_large.pdf",
        remove_y_label=True,
        remove_y_ticks=True,
    )
