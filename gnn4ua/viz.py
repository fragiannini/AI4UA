import torch_geometric as pyg
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from torch_geometric import seed_everything

from gnn4ua.utils import find_cluster_centroids
from sklearn.manifold import TSNE
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
import umap
import torch
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import os
import seaborn as sns

from .utils import find_k_hop_subgraph, pyg_edges_to_nx_lattice


def get_samples_to_plot(gnn, data, task_id, layer_id, figures_dir, all_labels,
                            concept_idx=0, n_concepts=5, random_state=42):
    seed_everything(random_state)
    y_pred, node_concepts, graph_concepts = gnn.forward(data)
    concepts = graph_concepts[layer_id]
    edge_indexes = pyg.utils.unbatch_edge_index(data.edge_index, data.batch)
    graph_sizes = torch.LongTensor([len(torch.unique(eidx.ravel())) for eidx in edge_indexes])

    concept_id = torch.argsort(gnn.dense_layers[0].weight, dim=1, descending=False)[task_id, concept_idx]
    correct_pred_mask = data.y[:, task_id] == (y_pred[:, task_id] > 0)

    reducer = umap.UMAP(metric='jaccard', random_state=random_state)
    reducer.fit(concepts[correct_pred_mask].detach().numpy(), y=data.y[correct_pred_mask, task_id])
    concepts2d = reducer.transform(concepts[correct_pred_mask].detach().numpy())

    saved_imgs = []
    samples_to_plot = []
    for task_state in [0, 1]:
        concept_mask = (concepts[:, concept_id] > 0) & \
                       (data.y[:, task_id] == task_state) & \
                       correct_pred_mask  # omission of these lattices is relevant
        concept_indexes = torch.argwhere(concept_mask).flatten()
        graph_sizes_sorted_idx = torch.argsort(graph_sizes[concept_indexes].long(), descending=False)
        concept_indexes_sorted = concept_indexes[graph_sizes_sorted_idx][:n_concepts]

        nc = 3 if task_state == 0 else 1
        if len(concept_indexes_sorted) < nc:
            print('Not enough concepts to plot')
            continue
        rnd_idx = np.random.RandomState(random_state).choice(len(concept_indexes_sorted), size=nc, replace=False)
        for graph_id in concept_indexes_sorted[rnd_idx]:
            edge_index = edge_indexes[graph_id]
            plt.figure(figsize=[2.5, 2.5])
            visualize_lattice(edge_index, node_size=300)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'task_{all_labels[task_id]}_layer_{layer_id}_concept_{int(concept_id)}.png'), transparent=True, dpi=20)
            plt.draw()
            saved_img = imread(os.path.join(figures_dir, f'task_{all_labels[task_id]}_layer_{layer_id}_concept_{int(concept_id)}.png'))
            saved_imgs.append(saved_img)
            samples_to_plot.append(concepts[graph_id])

    samples_to_plot = torch.stack(samples_to_plot)
    samples2d = reducer.transform(samples_to_plot.detach().numpy())
    return samples2d, saved_imgs, concepts2d, correct_pred_mask, concept_id


def visualize_concept_space(samples2d, saved_imgs, concepts2d, correct_pred_mask, data, task_id, full_names, cmap, cols, ax):
    legend_patches = []
    # plt.scatter(concepts2d[:, 0], concepts2d[:, 1], c=data.y[:, task_id], alpha=0.01, marker='.', s=5,
    #             edgecolors=None, clip_on=False, cmap=cmap)
    sns.kdeplot(x=concepts2d[:, 0], y=concepts2d[:, 1], hue=data.y[correct_pred_mask, task_id] > 0, fill=False,
                alpha=0.15, linewidth=0, clip_on=False, cmap=cmap)
    # sns.scatterplot(x=concepts2d[:, 0], y=concepts2d[:, 1], hue=data.y[correct_pred_mask, task_id] > 0, marker='.',
    #                 alpha=0.005, cmap=cmap)

    base_name = full_names[task_id].replace('_', ' ')
    for col, l_name in zip(cols, ['Non-' + base_name, base_name]):
        legend_patches.append(Patch(facecolor=col, label=l_name, alpha=0.4))
    # plt.scatter(samples2d[:, 0], samples2d[:, 1], s=1500, c=cols[0], alpha=0.3, clip_on=False)
    for sample_id in range(len(samples2d)):
        imagebox = OffsetImage(saved_imgs[sample_id], zoom=.7)
        ab = AnnotationBbox(imagebox, samples2d[sample_id], frameon=False)
        ax.add_artist(ab)
    plt.legend(handles=legend_patches, fontsize=12)


def visualize_lattice(edge_index, node_size=200, node_color='k', with_labels=False):
    digraph = pyg_edges_to_nx_lattice(edge_index)
    # try:
    #     nx.find_cycle(digraph, orientation='ignore')
    pos = graphviz_layout(digraph, prog='dot')
    nx.draw_networkx(digraph, pos=pos, node_size=node_size, node_color=node_color, with_labels=with_labels)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    # except:
    #     print('Lattice is trivial')


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


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
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.tight_layout()
    return
