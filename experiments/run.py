import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import os
from torch_geometric import seed_everything
import numpy as np

import sys
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
        metrics_dir = os.path.join(results_dir, f'task_{label_name}/metrics/')
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
                    HierarchicalGCN(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                    GCoRe(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                    BlackBoxGNN(data.x.shape[1], emb_size, data.y.shape[1], n_layers),
                ]

                for gnn in models:
                    model_path = os.path.join(model_dir, f'{gnn.__class__.__name__}_generalization_{generalization}_seed_{random_state}_temperature_{temperature}_embsize_{emb_size}.pt')
                    print(f'Running {gnn.__class__.__name__} on {label_name} ({generalization} generalization) [seed {state_id+1}/{len(random_states)}]')

                    if not os.path.exists(model_path):
                        # train model
                        optimizer = torch.optim.AdamW(gnn.parameters(), lr=learning_rate)
                        loss_form = torch.nn.CrossEntropyLoss(weight=1-data.y.float().mean(dim=0))
                        gnn.train()
                        for epoch in range(train_epochs):
                            optimizer.zero_grad()
                            y_pred, node_concepts, graph_concepts = gnn.forward(data)

                            # compute loss
                            main_loss = loss_form(y_pred[train_index], data.y[train_index].argmax(dim=1))
                            internal_loss = 0
                            if gnn.__class__.__name__ == 'HierarchicalGCN':
                                internal_loss = gnn.internal_loss(graph_concepts, data.y.argmax(dim=1), loss_form, train_index)
                            internal_loss = internal_loss * internal_loss_weight
                            loss = main_loss + internal_loss

                            loss.backward()
                            optimizer.step()

                            # monitor AUC
                            if epoch % 1 == 0:
                                train_auc = roc_auc_score(data.y[train_index].detach(), y_pred[train_index].detach())
                                train_accuracy = accuracy_score(data.y[train_index].detach(), y_pred[train_index].detach()>0.5)
                                print(f'Epoch {epoch}: loss={main_loss:.3f} [main={main_loss:.3f} internal={internal_loss:.3f}] train AUC={train_auc:.4f} train accuracy={train_accuracy:.4f}')

                        torch.save(gnn.state_dict(), model_path)

                    # get model predictions
                    gnn.load_state_dict(torch.load(model_path))
                    for param in gnn.parameters():
                        param.requires_grad = False
                    y_pred, node_concepts, graph_concepts = gnn.forward(data)

                    # evaluate predictions on test set and save results
                    test_auc = roc_auc_score(data.y[test_index].detach(), y_pred[test_index].detach())
                    print(f'Test accuracy: {test_auc:.4f}')
                    results.append([label_name, generalization, gnn.__class__.__name__, test_auc, train_auc, n_layers, temperature, emb_size, learning_rate, train_epochs, max_size_train, max_prob_train])
                    pd.DataFrame(results, columns=cols).to_csv(os.path.join(metrics_dir, 'auc.csv'))


if __name__ == '__main__':
    main()
