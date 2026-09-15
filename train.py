import argparse
from data import make_loaders
from models import create_models
import torch
from torch.nn import HuberLoss
from torch.utils.tensorboard import SummaryWriter
import os
import time
import tqdm
from pathlib import Path

def epoch_through_model(
    model,
    optimizer,
    data,
    epoch,
    writer,
    min_loss,
    device,
    model_save_dir,
    mode="train",
):
    assert mode in ["train", "val_when_training", "test_when_training", "val_only", "test_only"]

    # set model in right state
    if mode == "train":
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # initialize metrics for this epoch
    loss_list = []
    top_5 = 0
    top_10 = 0
    top_15 = 0
    top_20 = 0
    delta_5 = 0
    delta_10 = 0
    delta_15 = 0
    delta_20 = 0
    len_total = 0
    num_steps_done = epoch * len(data)

    # iterate over data
    pbar = tqdm.tqdm(data)
    for step, data_sample in enumerate(pbar):
        # put data on the device
        data_sample.to(device)

        
        out, loss, gt = model.get_loss_and_prediction_and_gt(data_sample)
        loss_list.append(loss.data)

        # calculate absolute difference between prediction and ground truth
        diff = torch.abs(out - gt)
        
        if epoch % 100 == 0 and epoch > 0 and mode == "train":
            print("before")
            print(f"{gt=}")
            print(f"{out=} {out.shape=}")
            print(f"{diff=}")

        # calculate metrics for this batch of data
        delta_5 += (diff < 5).sum()
        delta_10 += (diff < 10).sum()
        delta_15 += (diff < 15).sum()
        delta_20 += (diff < 20).sum()
        diff = (diff / (gt)) * 100
        top_5 += (diff < 5).sum()
        top_10 += (diff < 10).sum()
        top_15 += (diff < 15).sum()
        top_20 += (diff < 20).sum()
        len_total += len(diff)


        # backward pass if applicable
        if mode == "train":
            loss.backward()
            optimizer.step()

        # update tqdm bar
        pbar.set_description(f"Epoch: {epoch}")
        pbar.set_postfix({"loss": loss.detach().cpu().numpy().item()})

        # write to writer
        writer.add_scalar(f"loss/{mode}/step", loss, step + num_steps_done)

    # calculate mean loss across full epoch
    loss_mean = torch.mean(torch.tensor(loss_list))

    # save best performing model in validation
    if loss_mean < min_loss:
        min_loss = loss_mean
        if mode == "val_when_training":
            os.makedirs(f"{model_save_dir}", exist_ok=True)
            files = os.listdir(model_save_dir)
            for file in files:
                os.remove(f"{model_save_dir}/{file}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_mean,
                },
                f"{model_save_dir}/model_{min_loss}.pt",
            )

    # write epoch level metrics to writer
    writer.add_scalar(f"loss/{mode}/epoch", loss_mean, epoch)
    writer.add_scalar(f"loss/{mode}/min", min_loss, epoch)
    writer.add_scalar(f"loss/{mode}/top_5", top_5 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/top_10", top_10 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/top_15", top_15 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/top_20", top_20 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/delta_5", delta_5 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/delta_10", delta_10 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/delta_15", delta_15 * 100 / len_total, epoch)
    writer.add_scalar(f"loss/{mode}/delta_20", delta_20 * 100 / len_total, epoch)
    return min_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommonNoC entry point for training")
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
        "--model_type",
        type=str,
        choices=["graph_sage", "graph_sage_class_reg"],
        required=False,
        default="graph_sage",
        help="The type of model to use",
    )

    parser.add_argument(
        "--model_load_path",
        type=str,
        required=False,
        help="""If a model checkpoint will be loaded. Program will use this model
        to run.""",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=False,
        default=f"{os.getcwd()}/models/{time.time()}",
        help="""Directory in which best validating models will be saved. Models will
        be saved as model_{loss}.pt in this directory. If not specified, the directory will
        be a timestamped directory of the nature models/`timestamp` in CWD""",
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
        "--dropout",
        type=float,
        required=False,
        default=0.2,
        help="""Dropout to apply on the model when training. Must be between
        0 and 1""",
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
        "--batch_size", type=int, required=False, default=128, help="Batch size of data."
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
        "--lr",
        type=float,
        required=False,
        default=0.000001,
        help="""Learning rate of the training. Applicable only if actually training
        the model""",
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path from which the datasets will be loaded.",
    )

    parser.add_argument(
        "--use_weighting",
        type=bool,
        required=False,
        default=False,
        help="""If enabled, the train DataLoader is configured to have a sampling
        frequency that is tied to each sample's wclatency occurrence in the
        dataset. Frequency buckets are of size 10. Simply put, samples
        with more common latencies will be sampled fewer times. This
        allows the extreme latencies to be "seen" more frequently by
        the NN model. Defaults to False.""",
    )

    parser.add_argument(
        "--eval_model",
        action='store_true',
        help="""Model is only evaluated on validation and test sets,
        if set. Otherwise model is trained on the train set and
        evaluated on validation and test sets.""",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100000,
        help="Number of epochs to train the model for.",
    )

    # TODO: move this to its own function?
    args = parser.parse_args()
    list_N = args.N
    dataset_folder = args.dataset_folder
    ratio_data = args.ratio_data
    batch_size = args.batch_size
    limit = args.limit
    use_weighting = args.use_weighting
    model_type = args.model_type
    emb_size = args.emb_size
    num_layers = args.num_layers
    dropout = args.dropout
    limit = args.limit
    model_load_path = args.model_load_path
    eval_model = args.eval_model
    lr = args.lr
    epochs = args.epochs
    model_save_dir = args.model_save_dir

    # check wether to use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create data loaders
    train_loader, val_loader, test_loader = make_loaders(
        list_N=list_N,
        dataset_folder=dataset_folder,
        ratio_data=ratio_data,
        batch_size=batch_size,
        limit=limit,
        use_weighting=use_weighting,
    )

    # create model
    model = create_models(
        model_type=model_type,
        emb_size=emb_size,
        num_layers=num_layers,
        dropout=dropout,
        limit=limit,
    )
    # convert model to hetero model. See
    # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
    model.to_hetero(example=next(iter(train_loader)))
    # move model to device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
    writer = SummaryWriter()  # tensorboard logging

    # load checkpoint if defined
    if model_load_path is not None:
        # feed a sample to init the model shapes
        model(next(iter(train_loader)).to(device))
        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

    min_loss_val = float("inf")
    min_loss_test = float("inf")
    min_loss_train = float("inf")

    # kick off program
    for epoch in range(1 if eval_model else epochs):
        loss_list = []
        out_list = []
        gt_list = []
        # train model on entire train dataset
        
        min_loss_train = epoch_through_model(
            model=model,
            optimizer=optimizer,
            data=train_loader,
            epoch=epoch,
            writer=writer,
            min_loss=min_loss_train,
            device=device,
            model_save_dir=model_save_dir,
            mode="train"
        )
        # test model accuracy on validation dataset
        min_loss_val = epoch_through_model(
            model=model,
            optimizer=optimizer,
            data=val_loader,
            epoch=epoch,
            writer=writer,
            min_loss=min_loss_val,
            device=device,
            model_save_dir=model_save_dir,
            mode="val_only" if eval_model else "val_when_training",
        )
        # test model accuracy on test dataset
        min_loss_test = epoch_through_model(
            model=model,
            optimizer=optimizer,
            data=test_loader,
            epoch=epoch,
            writer=writer,
            min_loss=min_loss_test,
            device=device,
            model_save_dir=model_save_dir,
            mode="test_only" if eval_model else "test_when_training",
        )

    writer.flush()
    writer.close()
