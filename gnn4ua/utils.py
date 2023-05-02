import torch
import torch_geometric as pyg


def get_concepts_for_task(concepts, data, task_id):
    filtered_graphs = torch.argwhere(data.y[:, task_id]).ravel()
    batch_mask = torch.zeros_like(data.batch, dtype=torch.bool)
    batch_mask[filtered_graphs] = True
    return concepts[batch_mask]


def find_topk_nearest_examples(concepts, n_concepts, n_examples_per_concept):
    cluster_centroids, cluster_counts = torch.unique(concepts > 0.5, dim=0, return_counts=True)
    max_counts = torch.sort(cluster_counts, descending=True)[0][n_concepts]
    cluster_centroids_filtered = cluster_centroids[cluster_counts > max_counts]
    return torch.argsort(torch.cdist(cluster_centroids_filtered.float(), concepts))[:, :n_examples_per_concept]


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
