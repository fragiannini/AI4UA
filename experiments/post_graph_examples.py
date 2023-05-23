import torch
import os

from matplotlib.colors import ListedColormap
from torch_geometric import seed_everything
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import torch_geometric as pyg

sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe, HierarchicalGCN
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

                    figures_dir = os.path.join(results_dir, f'figures/{label_name}_{generalization}/')
                    os.makedirs(figures_dir, exist_ok=True)
                    summary_figures_dir = os.path.join(results_dir, f'figures/')
                    os.makedirs(summary_figures_dir, exist_ok=True)

                    # get model predictions
                    gnn.load_state_dict(torch.load(model_path))
                    for param in gnn.parameters():
                        param.requires_grad = False
                    y_pred, node_concepts, graph_concepts = gnn.forward(data)

                    plt.figure(figsize=[3, 4])
                    plt.title(f'HiGNN Classifier [L7]', fontsize=18)
                    sns.heatmap(gnn.dense_layers[0].weight.T, cmap='vlag', square=False)
                    plt.ylabel('Concept ID', fontsize=16)
                    plt.xlabel('Task label', fontsize=16)
                    plt.xticks(ticks=np.arange(0, 5)+0.5, labels=all_labels, rotation=0, fontsize=12)
                    yt = np.arange(0, emb_size)
                    plt.yticks(ticks=yt+0.5, rotation='horizontal', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_figures_dir, f'classifier_weights.png'), bbox_inches='tight')
                    plt.savefig(os.path.join(summary_figures_dir, f'classifier_weights.pdf'), bbox_inches='tight')
                    plt.show()

                    layer_ids = [7, 2]
                    cols = sns.color_palette("colorblind")[:2]
                    cmap = ListedColormap(sns.color_palette(cols).as_hex())
                    for layer_id in layer_ids:
                        for task_id in range(len(all_labels)):
                            for concept_idx in range(5):
                                samples2d, saved_imgs, concepts2d, correct_pred_mask, cid = get_samples_to_plot(gnn, data, task_id, layer_id, figures_dir, all_labels, concept_idx, n_concepts, random_state)

                                base_name = full_names[task_id].replace('_', ' ')
                                fig, ax = plt.subplots(figsize=(5, 3.5))
                                plt.title(f'Concept {cid.item()} [L{layer_id}]', fontsize=22)
                                visualize_concept_space(samples2d, saved_imgs, concepts2d, correct_pred_mask, data, task_id, full_names, cmap, cols, ax)
                                plt.xlabel('UMAP 1st PC', fontsize=14)
                                plt.ylabel('UMAP 2nd PC', fontsize=14)
                                sns.despine(offset=0)
                                plt.tight_layout()
                                plt.savefig(os.path.join(summary_figures_dir, f'task_{label_name}_{base_name}_layer_{layer_id}_cid_{cid.item()}.png'), bbox_inches='tight')
                                plt.savefig(os.path.join(summary_figures_dir, f'task_{label_name}_{base_name}_layer_{layer_id}_cid_{cid.item()}.pdf'), bbox_inches='tight')
                                plt.show()

                    break
                break


if __name__ == '__main__':
    main()
