import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import os
from torch_geometric import seed_everything

import sys
sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe
from gnn4ua.utils import get_concepts_for_task, find_topk_nearest_examples
from gnn4ua.viz import visualize_concept

# torch.autograd.set_detect_anomaly(True)
seed_everything(42)

def main():
    # hyperparameters
    random_state = 42
    dataset = 'samples_50_saved'
    temperature = 100
    label_names = ["Distributive", "Modular", "Meet_SemiDistributive", "Join_SemiDistributive", "SemiDistributive"]
    train_epochs = 50
    emb_size = 128
    learning_rate = 0.001
    n_concepts = 20
    n_examples_per_concept = 5
    n_hops_neighborhood = 4

    # we will save all results in this directory
    results_dir = f"results/igcore/"
    os.makedirs(results_dir, exist_ok=True)

    for label_name in label_names:
        model_dir = os.path.join(results_dir, f'dataset_{dataset}', f'task_{label_name}', f'temperature_{temperature}')
        os.makedirs(model_dir, exist_ok=True)
        figures_dir = os.path.join(results_dir, 'figures', f'task_{label_name}')
        os.makedirs(figures_dir, exist_ok=True)

        # load data and set up cross-validation
        data = load_data(dataset, label_name=label_name, root_dir='../gnn4ua/datasets/')
        n_classes = data.y.shape[1]
        y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in data.y])
        skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

        for fold, (train_index, test_index) in enumerate(skf.split(data.y, y_new)):
            # reset model weights for each fold
            models = [
                GCoRe(data.x.shape[1], emb_size, n_classes, n_layers=8),
                # BlackBoxGNN(data.x.shape[1], emb_size, n_classes, n_layers=8),
            ]
            for gnn in models:
                model_path = os.path.join(model_dir, f'{gnn.__class__.__name__}_fold_{fold}.pt')

                if not os.path.exists(model_path):
                    raise ValueError(f'Model {model_path} does not exist')

                # get model predictions
                gnn.load_state_dict(torch.load(model_path))
                for param in gnn.parameters():
                    param.requires_grad = False
                y_pred, concepts = gnn.forward(data)

                # visualize concept space before pooling
                concepts_filtered = get_concepts_for_task(concepts, data, task_id=0)
                topk_nearest_neighbors = find_topk_nearest_examples(concepts_filtered, n_concepts, n_examples_per_concept)

                # visualize concept using nearest neighbors
                for concept_id, nearest_neighbors in enumerate(topk_nearest_neighbors):
                    plt.figure(figsize=[n_examples_per_concept*3, 3])
                    visualize_concept(nearest_neighbors, data.edge_index, data.batch,
                                      n_examples_per_concept, n_hops_neighborhood,
                                      title=f'Task {label_name.replace("_", " ")} - Concept {concept_id}', font_size=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures_dir, f'fold_{fold}_concept_{concept_id}.png'))
                    plt.show()
            break


if __name__ == '__main__':
    main()
