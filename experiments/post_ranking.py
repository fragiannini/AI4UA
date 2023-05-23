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
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
sns.set_style('whitegrid', {'grid.linestyle': '--'})
plt.rcParams.update({
    "text.usetex": True,
})

def main():
    # hyperparameters
    random_states = np.random.RandomState(42).randint(0, 1000, size=3)
    dataset = 'samples_50_saved'
    temperature = 1
    full_names = ['Meet_SemiDistributive',
                 'SemiDistributive',
                 'Distributive',
                 'Join_SemiDistributive',
                 'Modular',]
    all_labels = ["MSD", "SD", "D", "JSD", "M"]
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

                    figures_dir = os.path.join(results_dir, f'figures/ranking/')
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

                    print('Plotting concept space')

                    layer_id = 7
                    task_id = 1
                    k_hop = 2


                    concepts = graph_concepts[layer_id]
                    edge_indexes = pyg.utils.unbatch_edge_index(data.edge_index, data.batch)
                    graph_sizes = torch.LongTensor([len(torch.unique(eidx.ravel())) for eidx in edge_indexes])

                    all_ids = [2, 4]
                    for task_id in [2, 4]:
                        excluded_labels = [i for i in all_ids if i != task_id]
                        saved_imgs = []
                        concept_ranking = torch.argsort(gnn.dense_layers[0].weight, dim=1, descending=False)[task_id]
                        for concept_id in concept_ranking[:5]:
                            reference_node_emb = torch.zeros(emb_size)
                            reference_node_emb[concept_id] = 1
                            node_mask = torch.sum(node_concepts[layer_id] == reference_node_emb, dim=1) == len(reference_node_emb)
                            node_ids = torch.argwhere(node_mask).ravel()
                            graph_ids = data.batch[node_ids].unique()

                            task_state = 1 if gnn.dense_layers[0].weight[task_id, concept_id] > 0 else 0
                            graph_mask = torch.argwhere(
                                (data.y[:, task_id] == task_state) #& (data.y[:, excluded_labels] == (1-task_state)).ravel()
                            ).ravel()
                            if len(graph_mask) == 0:
                                continue
                            graph_id = np.intersect1d(graph_ids, graph_mask)[0]

                            # node_id = torch.argwhere(torch.sum(node_concepts[layer_id][data.batch==graph_id] == reference_node_emb, dim=1) == len(reference_node_emb)).ravel()[0]
                            # n_nodes = len(edge_indexes[graph_id].ravel().unique())
                            # subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph([node_id], k_hop, edge_indexes[graph_id],
                            #                                                                   relabel_nodes=True, num_nodes=n_nodes)

                            digraph = pyg_edges_to_nx_lattice(edge_indexes[graph_id])

                            pos = graphviz_layout(digraph, prog='dot')
                            plt.figure(figsize=[2.5, 2.5])
                            nx.draw_networkx(digraph, pos=pos, node_size=500, width=2,
                                             with_labels=False, edge_color='w',
                                             node_color='w')
                            plt.gca().invert_yaxis()
                            plt.axis('off')
                            plt.tight_layout()
                            plt.savefig(os.path.join(figures_dir, f'task_{task_id}_layer_{layer_id}_concept_{int(concept_id)}.png'), transparent=True, dpi=20)
                            plt.show()
                            saved_img = imread(os.path.join(figures_dir, f'task_{task_id}_layer_{layer_id}_concept_{int(concept_id)}.png'))
                            saved_imgs.append(saved_img)

                        pad = 0.05
                        relevance_scores = gnn.dense_layers[0].weight[task_id, concept_ranking.flip(dims=(0,))].detach().numpy()[-5:]
                        positions = np.array([r-pad if r > 0 else r+pad for r in relevance_scores][::-1])

                        fig, ax = plt.subplots(figsize=(4, 3))

                        for sample_id in range(len(relevance_scores)):
                            imagebox = OffsetImage(saved_imgs[sample_id], zoom=.6)
                            xi = positions[sample_id]
                            ab = AnnotationBbox(imagebox, [len(relevance_scores)-1-sample_id, xi], frameon=False)
                            ax.add_artist(ab)
                            print(xi, len(relevance_scores)-1-sample_id)

                        plt.bar(range(len(relevance_scores)), relevance_scores)
                        plt.xlabel(f'Concept ID [{full_names[task_id]}]', fontsize=14)
                        plt.ylabel('Omission Relevance', fontsize=14)
                        plt.xticks(range(len(relevance_scores))[::-1], concept_ranking[:5].numpy())
                        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                        ax.xaxis.set_label_position('top')
                        plt.ylim([positions.min()-0.07, 0])
                        sns.despine()
                        plt.tight_layout()
                        plt.savefig(os.path.join(summary_figures_dir, f'task_{task_id}_layer_{layer_id}.png'), bbox_inches='tight')
                        plt.savefig(os.path.join(summary_figures_dir, f'task_{task_id}_layer_{layer_id}.pdf'), bbox_inches='tight')
                        plt.show()

                    break
                break


if __name__ == '__main__':
    main()
