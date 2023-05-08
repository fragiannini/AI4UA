import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import os

from sklearn.tree import DecisionTreeClassifier
from torch_geometric import seed_everything
import numpy as np

import sys

from gnn4ua.utils import completeness_score

sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe, HierarchicalGCN


# torch.autograd.set_detect_anomaly(True)
seed_everything(42)

def main():
    # hyperparameters
    random_states = np.random.RandomState(42).randint(0, 1000, size=3)
    dataset = 'samples_50_saved'
    temperature = 1
    label_names = ["Distributive", "Modular", "Meet_SemiDistributive", "Join_SemiDistributive", "SemiDistributive"]
    generalization_modes = ['weak', 'strong']
    train_epochs = 100
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
        metrics_dir = os.path.join(results_dir, f'metrics/')
        os.makedirs(metrics_dir, exist_ok=True)

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
                    BlackBoxGNN(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                    HierarchicalGCN(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                    GCoRe(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                ]

                for gnn in models:
                    model_path = os.path.join(model_dir, f'{gnn.__class__.__name__}_generalization_{generalization}_seed_{random_state}_temperature_{temperature}_embsize_{emb_size}.pt')
                    print(f'Running {gnn.__class__.__name__} on {label_name} ({generalization} generalization) [seed {state_id+1}/{len(random_states)}]')

                    if not os.path.exists(model_path):
                        raise ValueError(f'Model {model_path} does not exist')

                    # get model predictions
                    gnn.load_state_dict(torch.load(model_path))
                    for param in gnn.parameters():
                        param.requires_grad = False
                    y_pred, node_concepts, graph_concepts = gnn.forward(data)

                    if isinstance(gnn, HierarchicalGCN):
                        final_concepts = graph_concepts[-1]
                    else:
                        final_concepts = graph_concepts

                    clustering = True
                    k = 8
                    completeness = completeness_score(final_concepts.detach(), data.y, train_index, test_index, random_state, clustering, k)

                    print(f'Test accuracy: {completeness:.4f}')
                    results.append([label_name, generalization, gnn.__class__.__name__, completeness, None, n_layers, temperature,
                                    emb_size, learning_rate, train_epochs, max_size_train, max_prob_train])
                    pd.DataFrame(results, columns=cols).to_csv(os.path.join(metrics_dir, 'completeness.csv'))


if __name__ == '__main__':
    main()
