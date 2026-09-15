import pandas as pd
import seaborn as sns

def filter_df(df, *args, **kwargs):
    cond = None
    for key, val in kwargs.items():
        cond = cond & (df[key] == val) if cond is not None else (df[key] == val)
    return df[cond]

sns.set_theme()

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
# get df for qor
df_qor = filter_df(df, eval_gnn=False)

# create empty target df
target_df = pd.DataFrame()
# target_df = pd.DataFrame(columns=["N", "type", "max_it_s", "hw_type"])

# iterate over each df type
for df in (df_large_gnn_cpu, df_medium_gnn_cpu, df_large_gnn_gpu, df_medium_gnn_gpu, df_qor):
# for df in (df_large_gnn_gpu, df_large_gnn_cpu, df_medium_gnn_gpu, df_medium_gnn_cpu, df_qor):
    # iterate over each N
    for noc_size in df["N"].unique():
        # for this noc size, find maximum iterations achieved
        is_gnn = df["eval_gnn"].unique() == [True]
        is_large = (
            is_gnn
            and df["num_layers"].unique() == [10]
            and df["emb_size"].unique() == [256]
        )
        on_gpu = is_gnn and df["model_on_gpu"].unique() == [True]
        target_df = pd.concat(
            [
                target_df,
                pd.DataFrame(
                    {
                        "N": [noc_size],
                        "type": ["GNN-large"]
                        if (is_gnn and is_large)
                        else (["GNN-medium"] if (is_gnn and not is_large) else ["QoR"]),
                        "max_it_s": [filter_df(df, N=noc_size)["items_per_sec"].max()],
                        "hw_type": ["GPU"] if on_gpu else ["CPU"],
                    }
                ),
            ],
            ignore_index=True,
        )
        print()
        # print(noc_size, filter_df(df, N=noc_size))

target_df = target_df.sort_values(by=["N", "type"])
print(target_df)
line_plot = sns.lineplot(data=target_df, x="N", y="max_it_s", hue="type", style="hw_type")
fig = line_plot.get_figure()
fig.savefig("out.png") 