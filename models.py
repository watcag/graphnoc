import torch_geometric
from torch_geometric.nn import norm, to_hetero
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.utils import scatter
import torch
import math
from torch.nn import HuberLoss, CrossEntropyLoss, Sequential


# class GraphSage:
#     """
#     Creates an object for a GraphSAGE model
#     Parameters
#     ----------
#     emb_size: int
#         Size of the nodes' activations.
#     num_layers: int
#         Number of layers in the model.
#     out_size: int
#         Size of the output nodes.
#     dropout: float
#         Amount of dropout to apply
#     limit: int
#         Maximum output value that the model has to predict. Helps with
#         quicker convergence
#     """

#     def __init__(
#         self,
#         emb_size,
#         num_layers,
#         dropout,
#         limit,
#         model_type="graph_sage",
#         resolution=None,
#     ):
#         assert 0 <= dropout < 1
#         assert type(limit) == int
#         assert model_type in ["graph_sage", "graph_sage_class_reg"]

#         self.emb_size = emb_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.limit = limit
#         self.model_type = model_type
#         self.resolution = resolution

#         if self.model_type == "graph_sage":
#             self.model = models.GraphSAGE(
#                 -1,
#                 hidden_channels=emb_size,
#                 num_layers=num_layers,
#                 out_channels=3,
#                 dropout=dropout,
#                 norm=norm.LayerNorm(emb_size, mode="node"),
#             )
#         elif self.model_type == "graph_sage_class_reg":
#             self.model = GraphSageClassReg(
#                 emb_size=self.emb_size,
#                 num_layers=self.num_layers,
#                 dropout=self.dropout,
#                 limit=self.limit,
#                 resolution=self.resolution,
#             )

#     def quantize_model(self, repr_sample):
#         print(repr_sample)
#         self.disable_train_mode()
#         self.model.qconfig = torch.ao.quantization.get_default_qconfig("x86")

#         model = torch.ao.quantization.prepare(self.model)

#         model(repr_sample)

#         self.model = torch.ao.quantization.convert(model)

#     def compile(self):
#         torch_geometric.compile(self.model, dynamic=False)

#     def get_parameters(self):
#         """
#         Returns parameters of the model.
#         """
#         return self.model.parameters()

#     def to_hetero(self, example, aggr="sum"):
#         """
#         Converts a model to its heterogenous rendition. See
#         https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
#         for details
#         Parameters
#         ----------
#         example: HeteroData
#             An example data point for a heterogenous GNN.
#         aggr: str, optional
#             Type of aggregation to use in the heterogenous rendition. Defaults to `sum`.
#         """
#         self.model = to_hetero(
#             module=self.model, metadata=example.metadata(), aggr=aggr
#         )

#     def get_state_dict(self):
#         """
#         Returns the state_dict of the model
#         """
#         return self.model.state_dict()

#     def load_checkpoint(self, model_state_dict):
#         """
#         Loads up a saved checkpoint's parameters into the model
#         Parameters
#         ----------
#         model_state_dict: dict
#             Dictionary containing model parameters.
#         """
#         self.model.load_state_dict(model_state_dict)

#     def move_to_device(self, device):
#         """
#         Loads the model on chose device

#         Parameters
#         ----------
#         device: str
#             String based representation of CPU or GPU.
#         """
#         self.model.to(device)

#     def enable_train_mode(self):
#         self.model.train()

#     def disable_train_mode(self):
#         self.model.eval()

#     def forward(self, data_sample):
#         if self.model_type == "graph_sage":
#             out = self.forward_graph_sage(data_sample)
#         elif self.model_type == "graph_sage_class_reg":
#             out_class, out_reg = self.forward_graph_sage_class_reg(data_sample)

#     def forward_graph_sage_class_reg(self, data_sample):
#         return self.model(data_sample)

#     def forward_graph_sage(self, data_sample):
#         out = self.model(data_sample.x_dict, data_sample.edge_index_dict)

#         # get individual hetero outputs
#         out_port = scatter(out["port"], data_sample["port"].batch, dim=0, reduce="mean")
#         out_switch = scatter(
#             out["switch"], data_sample["switch"].batch, dim=0, reduce="mean"
#         )
#         out_trace = scatter(
#             out["trace"], data_sample["trace"].batch, dim=0, reduce="mean"
#         )

#         # generate m of (10**m)x + c
#         out_m_cat = torch.cat(
#             [
#                 out_port[:, 0].unsqueeze(-1),
#                 out_switch[:, 0].unsqueeze(-1),
#                 out_trace[:, 0].unsqueeze(-1),
#             ],
#             dim=-1,
#         )
#         out_m = torch.mean(out_m_cat, dim=-1)
#         out_m = torch.sigmoid(out_m) * math.log10(self.limit)

#         # generate x of (10**m)x + c
#         out_x_cat = torch.cat(
#             [
#                 out_port[:, 1].unsqueeze(-1),
#                 out_switch[:, 1].unsqueeze(-1),
#                 out_trace[:, 1].unsqueeze(-1),
#             ],
#             dim=-1,
#         )
#         out_x = torch.mean(out_x_cat, dim=-1)
#         out_x = torch.sigmoid(out_x)

#         # generate c of (10**m)x + c
#         out_c_cat = torch.cat(
#             [
#                 out_port[:, 2].unsqueeze(-1),
#                 out_switch[:, 2].unsqueeze(-1),
#                 out_trace[:, 2].unsqueeze(-1),
#             ],
#             dim=-1,
#         )
#         out_c = torch.mean(out_c_cat, dim=-1)
#         out_c = out_c

#         out = ((10**out_m) * out_x) + out_c

#         return out


class GraphSage(torch.nn.Module):
    def __init__(self, emb_size, num_layers, dropout, limit):
        super().__init__()
        assert 0 <= dropout < 1
        assert type(limit) == int

        self.emb_size = emb_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.limit = limit

        self.gnn = GraphSAGE(
            -1,
            hidden_channels=emb_size,
            num_layers=num_layers,
            out_channels=3,
            dropout=dropout,
            norm=norm.LayerNorm(emb_size, mode="node"),
        )

        self.criterion = HuberLoss(delta=1)  # define loss

    def to_hetero(self, example, aggr="sum"):
        """
        Converts a model to its heterogenous rendition. See
        https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        for details
        Parameters
        ----------
        example: HeteroData
            An example data point for a heterogenous GNN.
        aggr: str, optional
            Type of aggregation to use in the heterogenous rendition. Defaults to `sum`.
        """
        self.gnn = to_hetero(module=self.gnn, metadata=example.metadata(), aggr=aggr)

    def get_loss_and_prediction_and_gt(self, data_sample):
        out_model = self.forward(data_sample)
        out = out_model

        ground_truth = data_sample.y
        loss = self.criterion(out, ground_truth)

        print(out, ground_truth)
        return out, loss, ground_truth

    def forward(self, data_sample):
        out = self.gnn(data_sample.x_dict, data_sample.edge_index_dict)

        # get individual hetero outputs
        out_port = scatter(out["port"], data_sample["port"].batch, dim=0, reduce="mean")
        out_switch = scatter(
            out["switch"], data_sample["switch"].batch, dim=0, reduce="mean"
        )
        out_trace = scatter(
            out["trace"], data_sample["trace"].batch, dim=0, reduce="mean"
        )

        # generate m of (10**m)x + c
        out_m_cat = torch.cat(
            [
                out_port[:, 0].unsqueeze(-1),
                out_switch[:, 0].unsqueeze(-1),
                out_trace[:, 0].unsqueeze(-1),
            ],
            dim=-1,
        )
        out_m = torch.mean(out_m_cat, dim=-1)
        out_m = torch.sigmoid(out_m) * math.log10(self.limit)

        # generate x of (10**m)x + c
        out_x_cat = torch.cat(
            [
                out_port[:, 1].unsqueeze(-1),
                out_switch[:, 1].unsqueeze(-1),
                out_trace[:, 1].unsqueeze(-1),
            ],
            dim=-1,
        )
        out_x = torch.mean(out_x_cat, dim=-1)
        out_x = torch.sigmoid(out_x)

        # generate c of (10**m)x + c
        out_c_cat = torch.cat(
            [
                out_port[:, 2].unsqueeze(-1),
                out_switch[:, 2].unsqueeze(-1),
                out_trace[:, 2].unsqueeze(-1),
            ],
            dim=-1,
        )
        out_c = torch.mean(out_c_cat, dim=-1)
        out_c = out_c

        out = ((10**out_m) * out_x) + out_c

        return out


class GraphSageClassReg(torch.nn.Module):
    def __init__(self, emb_size, num_layers, dropout, limit, resolution=100):
        super().__init__()
        assert 0 <= dropout < 1
        assert type(limit) == int
        assert limit % resolution == 0

        self.emb_size = emb_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.limit = limit
        self.resolution = resolution
        self.num_classes = (limit // resolution) + 1
        print(f"{self.num_classes=}")

        self.gnn = GraphSAGE(
            -1,
            hidden_channels=emb_size,
            num_layers=num_layers,
            out_channels=emb_size,
            dropout=dropout,
            norm=norm.LayerNorm(emb_size, mode="node"),
        )

        self.classifier = Sequential(
            torch.nn.Linear(3 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, self.num_classes),
        )
        
        self.criterion_classifier = CrossEntropyLoss()

        self.regression = Sequential(
            torch.nn.Linear((3 * emb_size)+self.num_classes, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1),
        )
        self.criterion_regression = HuberLoss(delta=1)  # define loss


    def get_prediction(self, data_sample):
        # get model predictions
        out_class, out_reg = self.forward(data_sample)
        out_class_label = out_class.argmax(-1)
        pred_model = out_reg + (out_class_label * self.resolution)

        return pred_model, out_class, out_reg

    def get_loss_and_prediction_and_gt(self, data_sample):
        # get model predictions
        pred_model, out_class, out_reg = self.get_prediction(data_sample)
        
        # get ground truth
        ground_truth = data_sample.y
        class_gt = (ground_truth // self.resolution).long()
        reg_gt = ground_truth - (class_gt * self.resolution)

        # get loss
        class_loss = self.criterion_classifier(out_class, class_gt)
        reg_loss = self.criterion_regression(out_reg, reg_gt)
        loss = class_loss + (reg_loss/25)


        # print(out_class_label, class_label_gt)
        # print(out_class_label.shape, out_reg.shape)
        return pred_model, loss, ground_truth


    def to_hetero(self, example, aggr="sum"):
        """
        Converts a model to its heterogenous rendition. See
        https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        for details
        Parameters
        ----------
        example: HeteroData
            An example data point for a heterogenous GNN.
        aggr: str, optional
            Type of aggregation to use in the heterogenous rendition. Defaults to `sum`.
        """
        self.gnn = to_hetero(module=self.gnn, metadata=example.metadata(), aggr=aggr)

    def forward(self, data_sample):
        out = self.gnn(data_sample.x_dict, data_sample.edge_index_dict)

        # get individual hetero outputs
        out_port = scatter(out["port"], data_sample["port"].batch, dim=0, reduce="mean")
        out_switch = scatter(
            out["switch"], data_sample["switch"].batch, dim=0, reduce="mean"
        )
        out_trace = scatter(
            out["trace"], data_sample["trace"].batch, dim=0, reduce="mean"
        )

        # average them out to get a output of shape (batch, emb_size*3)
        out_gnn = torch.concat([out_port, out_switch, out_trace], dim=-1)


        # get classification and regression outputs.
        out_classifier = self.classifier(out_gnn)  # (batch, num_classes)

        # batch. num_classes+emb_size*3
        in_regression = torch.concat([out_gnn, out_classifier], dim=-1)

        out_regression = self.regression(in_regression).squeeze(-1).sigmoid()*self.resolution  # (batch, 1)

        return out_classifier, out_regression


def create_models(model_type, emb_size, num_layers, dropout, limit):
    """
    Creates a class object for the chosen model type
    Parameters
    ----------
    model_type: str
        String based identifier for a supported model type.
    emb_size: int
        Size of the nodes' activations.
    num_layers: int
        Number of layers in the model.
    out_size: int
        Size of the output nodes.
    dropout: float
        Amount of dropout to apply
    limit: int
        Maximum output value that the model has to predict. Helps with
        quicker convergence
    """

    assert model_type in ["graph_sage", "graph_sage_class_reg"]
    if model_type == "graph_sage":
        return GraphSage(
            emb_size=emb_size,
            num_layers=num_layers,
            dropout=dropout,
            limit=limit,
        )
    elif model_type == "graph_sage_class_reg":
        return GraphSageClassReg(
            emb_size=emb_size,
            num_layers=num_layers,
            dropout=dropout,
            limit=limit,
            resolution=100,
        )
