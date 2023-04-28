import abc

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, Sequential
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
    def __init__(self, in_features, emb_size, n_classes, n_layers=8):
        super(GCoRe, self).__init__(in_features, emb_size, n_classes, n_layers)

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
                    LeakyReLU(inplace=True),
                ]
            ))
            in_size = emb_size

        # for the last graph layer we don't want to apply an activation but rather a pooling layer
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

        # TODO: here we want to add a gumbel softmax layer to extract concepts
        #self.gumbel = F.gumbel_softmax()
        # last, we apply a set of linear layers to get the final prediction
        self.dense_layers = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, n_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, layer in enumerate(self.graph_layers):
            x = layer(x, edge_index)

        
        c = F.gumbel_softmax(x)

        x = global_add_pool(c, batch)

        x = self.dense_layers(x)

        return torch.softmax(x, dim=-1), c


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
                    LeakyReLU(inplace=True),
                ]
            ))
            in_size = emb_size

        # for the last graph layer we don't want to apply an activation but rather a pooling layer
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

        # last, we apply a set of linear layers to get the final prediction
        self.dense_layers = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, n_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, layer in enumerate(self.graph_layers):
            x = layer(x, edge_index)

        x = global_add_pool(x, batch)
        x = self.dense_layers(x)

        return torch.softmax(x, dim=-1)
