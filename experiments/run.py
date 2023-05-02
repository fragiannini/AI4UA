from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import os
from torch_geometric import seed_everything

import sys
sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe


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

    # we will save all results in this directory
    results_dir = f"results/igcore/"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    cols = ['rules', 'accuracy', 'fold', 'model', 'task', 'dataset', 'temperature']
    for label_name in label_names:
        model_dir = os.path.join(results_dir, f'dataset_{dataset}', f'task_{label_name}', f'temperature_{temperature}')
        os.makedirs(model_dir, exist_ok=True)

        # load data and set up cross-validation
        data = load_data(dataset, label_name=label_name, root_dir='../gnn4ua/datasets/')
        n_classes = data.y.shape[1]
        y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in data.y])
        skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

        for fold, (train_index, test_index) in enumerate(skf.split(data.y, y_new)):
            # reset model weights for each fold
            models = [
                GCoRe(data.x.shape[1], emb_size, n_classes, n_layers=8),
                BlackBoxGNN(data.x.shape[1], emb_size, n_classes, n_layers=8),
            ]
            for gnn in models:
                model_path = os.path.join(model_dir, f'{gnn.__class__.__name__}_fold_{fold}.pt')

                if not os.path.exists(model_path):
                    # train model
                    optimizer = torch.optim.AdamW(gnn.parameters(), lr=learning_rate)
                    loss_form = torch.nn.CrossEntropyLoss(weight=1-data.y.float().mean(dim=0))
                    gnn.train()
                    for epoch in range(train_epochs):
                        optimizer.zero_grad()
                        y_pred, concepts = gnn.forward(data)
                        loss = loss_form(y_pred[train_index], data.y[train_index].argmax(dim=1))
                        loss.backward()
                        optimizer.step()

                        # monitor AUC
                        if epoch % 1 == 0:
                            train_auc = roc_auc_score(data.y[train_index].detach(), y_pred[train_index].detach())
                            train_accuracy = accuracy_score(data.y[train_index].detach(), y_pred[train_index].detach()>0.5)
                            print(f'Epoch {epoch}: loss {loss:.4f} train AUC: {train_auc:.4f} train accuracy: {train_accuracy:.4f}')

                    torch.save(gnn.state_dict(), model_path)

                # get model predictions
                gnn.load_state_dict(torch.load(model_path))
                for param in gnn.parameters():
                    param.requires_grad = False
                y_pred, concepts = gnn.forward(data)

                # evaluate predictions on test set and save results
                test_auc = roc_auc_score(data.y[test_index].detach(), y_pred[test_index].detach())
                print(f'Test accuracy: {test_auc:.4f}')
                results.append(['', test_auc, fold, gnn.__class__.__name__, label_name, dataset, temperature])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'metrics.csv'))


if __name__ == '__main__':
    main()
