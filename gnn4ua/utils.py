import torch
import torch_geometric as pyg
from sklearn.metrics import roc_auc_score, accuracy_score
import torch

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def completeness_score(embeddings, y, train_index, test_index,
                       random_state=42, clustering=False, k=8):
    # find concepts
    if clustering:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(embeddings[train_index])
        concepts = kmeans.predict(embeddings).reshape(-1, 1)
    else:
        concepts = (embeddings > 0.5).detach()

    # compute completeness scores
    classifier = DecisionTreeClassifier(random_state=random_state)
    classifier.fit(concepts[train_index], y[train_index].detach())
    y_pred = classifier.predict(concepts)

    # evaluate predictions on test set and save results
    return roc_auc_score(y[test_index].detach(), y_pred[test_index]), y_pred


def find_cluster_centroids(concepts):
    cluster_centroids, cluster_counts = torch.unique(concepts > 0.5, dim=0, return_counts=True)
    cluster_counts_sorted_idx = torch.argsort(cluster_counts, descending=True)
    return cluster_centroids[cluster_counts_sorted_idx], cluster_counts[cluster_counts_sorted_idx]


def find_small_concepts(data, concepts, task_id, max_size=8, max_examples=2):
    cluster_centroids, cluster_counts = find_cluster_centroids(concepts)
    edge_indexes = pyg.utils.unbatch_edge_index(data.edge_index, data.batch)
    graph_sizes = torch.LongTensor([len(torch.unique(eidx.ravel())) for eidx in edge_indexes])

    small_edge_indexes = {}
    for cluster_id, cluster_centroid in enumerate(cluster_centroids):
        small_edge_indexes[f'concept_{cluster_id}'] = {}
        concept_mask = torch.sum(concepts == cluster_centroid, dim=1) == concepts.shape[1]
        graph_ids = torch.unique(data.batch[concept_mask])
        graph_sizes_sort_idx = torch.argsort(graph_sizes[graph_ids], descending=False)
        graph_sizes_sort = graph_sizes[graph_ids][graph_sizes_sort_idx]
        graph_ids_sorted = graph_ids[graph_sizes_sort_idx]

        for size in range(1, max_size):
            small_edge_indexes[f'concept_{cluster_id}'][f'size_{size}'] = []
            graph_ids_size = graph_ids_sorted[graph_sizes_sort == size]
            examples_count = 0
            for example_id, graph_id in enumerate(graph_ids_size):
                if data.y[graph_id, task_id] == 0:
                    continue
                examples_count += 1
                if examples_count > max_examples:
                    break

                small_edge_indexes[f'concept_{cluster_id}'][f'size_{size}'].append(edge_indexes[graph_id])
    return small_edge_indexes


def pyg_edges_to_nx_lattice(edge_index):
    data_subset = pyg.data.Data(edge_index=edge_index)
    graph = pyg.utils.to_networkx(data_subset, to_undirected=True, remove_self_loops=True)
    digraph = graph.to_directed()
    backward_edges = [edge for edge in digraph.edges if edge[0] > edge[1]]
    digraph.remove_edges_from(backward_edges)
    return digraph


def find_topk_nearest_examples(data, concepts, task_id, n_concepts, n_examples_per_concept):
    # find centroids of largest clusters
    cluster_centroids, cluster_counts = torch.unique(concepts > 0.5, dim=0, return_counts=True)
    n_concepts = min([n_concepts, len(cluster_centroids)-1])
    max_counts = torch.sort(cluster_counts, descending=True)[0][n_concepts]
    cluster_centroids_filtered = cluster_centroids[cluster_counts > max_counts]

    # find top-k nearest neighbors for each cluster centroid
    closest_samples = torch.argsort(torch.cdist(cluster_centroids_filtered.float(), concepts))
    top_k_samples = []
    for centroid_id in range(len(cluster_centroids_filtered)):
        topk_centroid = []
        for sample_id in closest_samples[centroid_id]:
            if data.y[data.batch[sample_id], task_id] == 1:
                topk_centroid.append(sample_id)
            if len(topk_centroid) == n_examples_per_concept:
                break
        top_k_samples.append(torch.LongTensor(topk_centroid))
    return torch.stack(top_k_samples)


def find_k_hop_subgraph(node_idx, edge_index, batch_index, n_hops_neighborhood=1):
    # get p-hop neighborhood of the given node and return the corresponding subgraph
    edge_indexes = pyg.utils.unbatch_edge_index(edge_index, batch_index)
    batch_id = batch_index[node_idx]
    new_node_id = torch.argwhere(torch.argwhere(batch_index == batch_id).ravel() == node_idx).ravel()
    edge_index_batch = edge_indexes[batch_id]
    edge_index_batch_un = pyg.utils.to_undirected(edge_index_batch)
    subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph(new_node_id, num_hops=n_hops_neighborhood,
                                                                      edge_index=edge_index_batch_un,
                                                                      relabel_nodes=True)

    # eliminate all edges in edge_index that are not in edge_index_batch
    final_edge_index = []
    for e in edge_index.T:
        for e2 in edge_index_batch.T:
            if torch.all(e == e2):
                final_edge_index.append(e)
    final_edge_index = torch.stack(final_edge_index).T
    final_edge_index = pyg.utils.remove_self_loops(final_edge_index)[0]
    return pyg.utils.remove_isolated_nodes(final_edge_index)[0]
