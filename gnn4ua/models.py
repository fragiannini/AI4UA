import abc

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, Sequential, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch.nn import Linear, ModuleList, LeakyReLU


class GraphNet(torch.nn.Module):
    def __init__(self, in_features, emb_size, n_classes, n_layers=8):
        super(GraphNet, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.n_layers = n_layers


    @abc.abstractmethod
    def forward(self, data):
        raise NotImplementedError


class GCoRe(GraphNet):
    def __init__(self, in_features, emb_size, n_classes, n_layers=8, temperature=1.0, hard=True):
        super(GCoRe, self).__init__(in_features, emb_size, n_classes, n_layers)

        self.temperature = temperature
        self.hard = hard
        in_size = in_features
        self.graph_layers = ModuleList()
        for i in range(self.n_layers-1):
            self.graph_layers.append(Sequential(
                'x, edge_index', [
                    (GINConv(
                        torch.nn.Sequential(
                            Linear(in_size, emb_size),
                            LeakyReLU(),
                            Linear(emb_size, emb_size)
                        )
                    ), 'x, edge_index -> x'),
                    BatchNorm(emb_size),
                ]
            ))
            in_size = emb_size

        # last, we apply a set of linear layers to get the final prediction
        self.dense_layers = torch.nn.Sequential(
            # Linear(emb_size, emb_size),
            # torch.nn.LeakyReLU(),
            Linear(emb_size, n_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        n_layers = len(self.graph_layers)
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, edge_index)
            if i < n_layers - 1:
                x = F.leaky_relu(x)

        node_concepts = F.gumbel_softmax(x, tau=self.temperature, hard=self.hard)
        graph_concepts = global_max_pool(node_concepts, batch)
        x = self.dense_layers(graph_concepts)
        return x, node_concepts, graph_concepts


class HierarchicalGCN(GraphNet):
    def __init__(self, in_features, emb_size, n_classes, n_layers=8, temperature=1.0, hard=True):
        super(HierarchicalGCN, self).__init__(in_features, emb_size, n_classes, n_layers)

        self.temperature = temperature
        self.hard = hard
        in_size = in_features
        self.graph_layers = ModuleList()
        self.internal_classification_layers = ModuleList()
        for i in range(self.n_layers):
            self.graph_layers.append(Sequential(
                'x, edge_index', [
                    (GINConv(
                        torch.nn.Sequential(
                            Linear(in_size, emb_size),
                            LeakyReLU(),
                            Linear(emb_size, emb_size)
                        )
                    ), 'x, edge_index -> x'),
                    BatchNorm(emb_size),
                ]
            ))
            if i < self.n_layers - 1:
                self.internal_classification_layers.append(Linear(emb_size, n_classes))
            in_size = emb_size

        # last, we apply a set of linear layers to get the final prediction
        self.dense_layers = torch.nn.Sequential(
            # Linear(emb_size, emb_size),
            # torch.nn.LeakyReLU(),
            Linear(emb_size, n_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        node_concepts = []
        graph_concepts = []
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, edge_index)
            node_concept = F.gumbel_softmax(x, tau=self.temperature, hard=self.hard)
            graph_concept = global_max_pool(node_concept, batch)
            node_concepts.append(node_concept)
            graph_concepts.append(graph_concept)
            x = F.leaky_relu(x)

        x = self.dense_layers(graph_concept)
        return x, node_concepts, graph_concepts

    def internal_loss(self, graph_concepts, y, loss_form, data_index):
        loss = 0
        for graph_concept, classification_layer in zip(graph_concepts[:-1], self.internal_classification_layers):
            y_pred = classification_layer(graph_concept[data_index])
            loss += loss_form(y_pred, y[data_index])
        return loss

class BlackBoxGNN(GraphNet):
    def __init__(self, in_features, emb_size, n_classes, n_layers=8):
        super(BlackBoxGNN, self).__init__(in_features, emb_size, n_classes, n_layers)

        in_size = in_features
        self.graph_layers = ModuleList()
        for i in range(self.n_layers-1):
            self.graph_layers.append(Sequential(
                'x, edge_index', [
                    (GINConv(
                        torch.nn.Sequential(
                            Linear(in_size, emb_size),
                            LeakyReLU(),
                            Linear(emb_size, emb_size)
                        )
                    ), 'x, edge_index -> x'),
                    BatchNorm(emb_size),
                ]
            ))
            in_size = emb_size

        # last, we apply a set of linear layers to get the final prediction
        self.dense_layers = torch.nn.Sequential(
            # Linear(emb_size, emb_size),
            # torch.nn.LeakyReLU(),
            Linear(emb_size, n_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        n_layers = len(self.graph_layers)
        for i, layer in enumerate(self.graph_layers):
            x = layer(x, edge_index)
            if i < n_layers - 1:
                x = F.leaky_relu(x)

        cemb = global_max_pool(x, batch)
        x = self.dense_layers(cemb)
        return x, None, cemb
