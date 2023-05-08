import torch
import os
from torch_geometric import seed_everything
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from gnn4ua.utils import find_cluster_centroids, find_small_concepts
from sklearn.manifold import TSNE
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from gnn4ua.viz import visualize_lattice

sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe, HierarchicalGCN


# torch.autograd.set_detect_anomaly(True)
seed_everything(42)

# viz parameters
n_examples_per_concept = 5
n_concepts = 20
n_hops_neighborhood = 8

def main():
    # hyperparameters
    random_states = np.random.RandomState(42).randint(0, 1000, size=3)
    dataset = 'samples_50_saved'
    temperature = 1
    label_names = ["Distributive", "Modular", "Meet_SemiDistributive", "Join_SemiDistributive", "SemiDistributive"]
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

                    figures_dir = os.path.join(results_dir, f'figures/{label_name}_{generalization}/')
                    os.makedirs(figures_dir, exist_ok=True)
                    summary_figures_dir = os.path.join(results_dir, f'figures/')
                    os.makedirs(summary_figures_dir, exist_ok=True)

                    # get model predictions
                    gnn.load_state_dict(torch.load(model_path))
                    for param in gnn.parameters():
                        param.requires_grad = False
                    _, node_concepts, graph_concepts = gnn.forward(data)

                    layer_ids = [2, 7]
                    task_ids = [0, 1]
                    for layer_id in layer_ids:
                        for task_id in task_ids:

                            # find small subgraphs closest to the centroid of each concept
                            concepts = node_concepts[layer_id]
                            edge_indexes = find_small_concepts(data, concepts, task_id, max_size=8, max_examples=1)

                            imgs = {}
                            for concept_name, concept_examples in edge_indexes.items():
                                for size_name, concept_examples_size in concept_examples.items():
                                    for edge_index_example in concept_examples_size:
                                        concept_id = concept_name.split('_')[1]
                                        size = size_name.split('_')[1]
                                        print(f'Plotting task {label_name} - concept {concept_id} - layer {layer_id} - size {size} - label {task_id}')

                                        plt.figure(figsize=[2.5, 2.5])
                                        visualize_lattice(edge_index_example, node_size=300)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(figures_dir, f'task_{label_name}_type_{task_id}_layer_{layer_id}_concept_{concept_id}.png'), transparent=True, dpi=20)
                                        plt.show()

                                        imgs[concept_id] = imread(os.path.join(figures_dir, f'task_{label_name}_type_{task_id}_layer_{layer_id}_concept_{concept_id}.png'))
                                        break

                            cluster_centroids, cluster_counts = find_cluster_centroids(concepts)
                            tsne = TSNE(n_components=2, perplexity=len(cluster_centroids)-1).fit_transform(cluster_centroids)

                            fig, ax = plt.subplots(figsize=(6,4))
                            label = label_name.replace('_', ' ') if task_id==1 else 'Non-' + label_name.replace('_', ' ')
                            plt.suptitle(f'{label} concepts (layer {layer_id})', fontsize=14)
                            sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], s=cluster_counts.numpy()/10, alpha=0.5, clip_on=False)

                            for concept_id in range(len(cluster_centroids)):
                                imagebox = OffsetImage(imgs[f'{concept_id}'], zoom=.8)
                                # Container for the imagebox referring to a specific position *xy*.
                                ab = AnnotationBbox(imagebox, tsne[concept_id], frameon=False)
                                ax.add_artist(ab)

                            plt.xlabel('t-SNE 1st component')
                            plt.ylabel('t-SNE 2nd component')
                            sns.despine(offset=20)
                            plt.tight_layout()
                            plt.savefig(os.path.join(summary_figures_dir, f'task_{label_name}_type_{task_id}_layer_{layer_id}.png'))
                            plt.savefig(os.path.join(summary_figures_dir, f'task_{label_name}_type_{task_id}_layer_{layer_id}.pdf'))
                            plt.show()

                    break
                break


if __name__ == '__main__':
    main()
