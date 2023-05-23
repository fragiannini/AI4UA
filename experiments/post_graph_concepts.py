import torch
import os

from matplotlib.colors import ListedColormap
from torch_geometric import seed_everything
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import torch_geometric as pyg
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe, HierarchicalGCN
from gnn4ua.utils import pyg_edges_to_nx_lattice
from gnn4ua.viz import visualize_lattice, get_samples_to_plot, visualize_concept_space

# torch.autograd.set_detect_anomaly(True)
seed_everything(42)

# viz parameters
n_examples_per_concept = 5
n_concepts = 20
n_hops_neighborhood = 8
plt.rcParams.update({
    "text.usetex": True,
})

def main():
    # hyperparameters
    random_states = np.random.RandomState(42).randint(0, 1000, size=3)
    dataset = 'samples_50_saved'
    temperature = 1
    full_names = ['Meet_SemiDistributive',
                 'Distributive',
                 'Join_SemiDistributive',
                 'Modular',
                 'SemiDistributive']
    all_labels = ["MSD", "D", "JSD", "M", "SD"]
    label_names = ["multilabel"]
    generalization_modes = ['weak']
    train_epochs = 200
    emb_size = 16
    learning_rate = 0.005
    max_size_train = 8
    max_prob_train = 0.8
    n_layers = 8
    internal_loss_weight = 0.1

    # we will save all results in this directory
    results_dir = f"results/"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    cols = ['task', 'generalization', 'model', 'test_auc', 'train_auc', 'n_layers', 'temperature', 'emb_size', 'learning_rate', 'train_epochs', 'max_size_train', 'max_prob_train']
    for label_name in label_names:
        model_dir = os.path.join(results_dir, f'task_{label_name}/models/')
        os.makedirs(model_dir, exist_ok=True)
        figures_dir = os.path.join(results_dir, f'task_{label_name}/figures/')
        os.makedirs(figures_dir, exist_ok=True)

        for generalization in generalization_modes:
            for state_id, random_state in enumerate(random_states):

                # load data and set up cross-validation
                data = load_data(dataset, label_name=label_name, root_dir='../gnn4ua/datasets/',
                                 generalization=generalization, random_state=random_state,
                                 max_size_train=max_size_train, max_prob_train=max_prob_train)
                train_index, test_index = data.train_mask, data.test_mask
                print(sum(train_index), sum(test_index))

                # reset model weights for each fold
                models = [
                    HierarchicalGCN(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                    # GCoRe(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                    # BlackBoxGNN(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                ]

                for gnn in models:
                    model_path = os.path.join(model_dir, f'{gnn.__class__.__name__}_generalization_{generalization}_seed_{random_state}_temperature_{temperature}_embsize_{emb_size}.pt')
                    print(f'Running {gnn.__class__.__name__} on {label_name} ({generalization} generalization) [seed {state_id+1}/{len(random_states)}]')

                    if not os.path.exists(model_path):
                        raise ValueError(f'Model {model_path} does not exist')

                    figures_dir = os.path.join(results_dir, f'figures/manual/')
                    os.makedirs(figures_dir, exist_ok=True)
                    summary_figures_dir = os.path.join(results_dir, f'figures/')
                    os.makedirs(summary_figures_dir, exist_ok=True)

                    # get model predictions
                    gnn.load_state_dict(torch.load(model_path))
                    for param in gnn.parameters():
                        param.requires_grad = False
                    y_pred, node_concepts, graph_concepts = gnn.forward(data)

                    layer_ids = [2]
                    cols = sns.color_palette("colorblind")[:2]
                    cmap = ListedColormap(sns.color_palette(cols).as_hex())
                    edge_indexes = pyg.utils.unbatch_edge_index(data.edge_index, data.batch)
                    graph_sizes = torch.LongTensor([len(torch.unique(eidx.ravel())) for eidx in edge_indexes])
                    cols = sns.color_palette("colorblind")[:2]

                    reference_graph_id = 9527
                    for layer_id in layer_ids:
                        # get reference graph
                        reference_graph_concept = graph_concepts[layer_id][reference_graph_id]
                        most_similar_graphs = torch.argwhere(torch.sum(graph_concepts[layer_id] == reference_graph_concept, dim=1) == len(reference_graph_concept))
                        most_similar_graphs = list(most_similar_graphs.squeeze().numpy())
                        graph_ids.append(reference_graph_id)
                        for graph_id in most_similar_graphs:
                            digraph = pyg_edges_to_nx_lattice(edge_indexes[graph_id])

                            pos = graphviz_layout(digraph, prog='dot')
                            plt.figure(figsize=[2.5, 2.5])
                            plt.title(f'Concept labels: {str(list(reference_graph_concept.int().numpy())).replace(", ", "")}')
                            nx.draw_networkx(digraph, pos=pos, node_size=200, width=2,
                                             with_labels=False, edge_color='k',
                                             node_color='k')
                            plt.gca().invert_yaxis()
                            plt.axis('off')
                            plt.tight_layout()
                            plt.savefig(os.path.join(figures_dir, f'layer_{layer_id}_graph_{graph_id}.png'), dpi=300)
                            plt.savefig(os.path.join(figures_dir, f'layer_{layer_id}_graph_{graph_id}.pdf'), dpi=300)
                            plt.show()

                        # nodes
                        node_emb = node_concepts[layer_id][data.batch == reference_graph_id]
                        reference_node_emb = node_emb[0]
                        node_mask = torch.sum(node_concepts[layer_id] == reference_node_emb, dim=1) == len(reference_node_emb)
                        rnd_idx = np.random.RandomState(random_state).choice(np.arange(1000), size=3, replace=False)
                        node_ids = torch.argwhere(node_mask)[rnd_idx]
                        graph_ids = list(data.batch[node_ids].detach().numpy().ravel())
                        graph_ids.append(reference_graph_id)
                        for graph_id in graph_ids:
                            for k_hop in [1, 2]:
                                node_id = torch.argwhere(torch.sum(node_concepts[layer_id][data.batch==graph_id] == reference_node_emb, dim=1) == len(reference_node_emb)).ravel()[0]
                                n_nodes = len(edge_indexes[graph_id].ravel().unique())
                                subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph([node_id], k_hop, edge_indexes[graph_id],
                                                                                                  relabel_nodes=True, num_nodes=n_nodes)

                                data_subset = pyg.data.Data(edge_index=edge_indexes[graph_id])
                                digraph = pyg.utils.to_networkx(data_subset, to_undirected=False, remove_self_loops=False)
                                options = [cols[1], 'k']
                                colors = []
                                backward_edges = []
                                node_colors = ['k'] * len(digraph.nodes())
                                for oldeid, edge in enumerate(digraph.edges):
                                    if edge[0] > edge[1]:
                                        backward_edges.append(edge)
                                    else:
                                        if edge_mask[oldeid] == True:
                                            colors.append(options[0])
                                        else:
                                            colors.append(options[1])

                                digraph.remove_edges_from(backward_edges)
                                node_colors[node_id] = cols[1]

                                pos = graphviz_layout(digraph, prog='dot')
                                plt.figure(figsize=[2.5, 2.5])
                                plt.title(f'Concept labels: {str(list(reference_node_emb.int().numpy())).replace(", ", "")}')
                                nx.draw_networkx(digraph, pos=pos, node_size=200, width=2,
                                                 with_labels=False, edge_color=colors,
                                                 node_color=node_colors)
                                plt.gca().invert_yaxis()
                                plt.axis('off')
                                plt.tight_layout()
                                plt.savefig(os.path.join(figures_dir, f'layer_{layer_id}_khop_{k_hop}_graph_{graph_id}_node_{node_id}.png'), dpi=300)
                                plt.savefig(os.path.join(figures_dir, f'layer_{layer_id}_khop_{k_hop}_graph_{graph_id}_node_{node_id}.pdf'), dpi=300)
                                plt.show()



                    break
                break


if __name__ == '__main__':
    main()
