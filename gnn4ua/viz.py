import torch_geometric as pyg
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from .utils import find_k_hop_subgraph


def visualize_concept(nearest_neighbors_index, edge_index, batch_index,
                      n_examples_per_concept=5, n_hops_neighborhood=1, title='',
                      font_size=20, node_size=200, node_color='k', with_labels=False):
    plt.suptitle(title, fontsize=font_size)
    for example_id, node_idx in enumerate(nearest_neighbors_index):
        edge_index_subgraph = find_k_hop_subgraph(node_idx, edge_index, batch_index, n_hops_neighborhood)

        # transform to networkx graph and plot
        data_subset = pyg.data.Data(edge_index=edge_index_subgraph)
        digraph = pyg.utils.to_networkx(data_subset, to_undirected=False, remove_self_loops=True)
        pos = graphviz_layout(digraph, prog='dot')
        plt.subplot(1, n_examples_per_concept, example_id + 1)
        nx.draw_networkx(digraph, pos=pos, node_size=node_size, node_color=node_color, with_labels=with_labels)
        plt.axis('off')
        plt.tight_layout()
    return
