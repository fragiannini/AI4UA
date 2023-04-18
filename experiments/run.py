from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import os
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm
from torch.nn import Linear

from gnn4ua.datasets.loader import load_data

# torch.autograd.set_detect_anomaly(True)


class GCoRe(torch.nn.Module):
    def __init__(self, in_features, emb_size, n_classes):
        super(GCoRe, self).__init__()
        self.nn1 = torch.nn.Sequential(
            Linear(in_features, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn2 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn3 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn4 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn5 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn6 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn7 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.nn8 = torch.nn.Sequential(
            Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            Linear(emb_size, emb_size),
        )
        self.conv1 = GINConv(self.nn1)
        self.conv2 = GINConv(self.nn2)
        self.conv3 = GINConv(self.nn3)
        self.conv4 = GINConv(self.nn4)
        self.conv5 = GINConv(self.nn5)
        self.conv6 = GINConv(self.nn6)
        self.conv7 = GINConv(self.nn7)
        self.conv8 = GINConv(self.nn8)
        self.bn1 = BatchNorm(emb_size)
        self.bn2 = BatchNorm(emb_size)
        self.bn3 = BatchNorm(emb_size)
        self.bn4 = BatchNorm(emb_size)
        self.bn5 = BatchNorm(emb_size)
        self.bn6 = BatchNorm(emb_size)
        self.bn7 = BatchNorm(emb_size)
        self.bn8 = BatchNorm(emb_size)
        self.linear1 = Linear(emb_size, emb_size)
        self.linear2 = Linear(emb_size, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.leaky_relu(x)

        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.leaky_relu(x)

        x = self.conv6(x, edge_index)
        x = self.bn6(x)
        x = F.leaky_relu(x)

        x = self.conv7(x, edge_index)
        x = self.bn7(x)
        x = F.leaky_relu(x)

        x = self.conv8(x, edge_index)
        x = self.bn8(x)

        x = global_add_pool(x, batch)

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)

        return torch.softmax(x, dim=-1)


def main():
    # hyperparameters
    random_state = 42
    datasets = ['samples_50_saved']
    temperatures = [100]
    label_names = ["Distributive", "Modular", "Meet_SemiDistributive", "Join_SemiDistributive", "SemiDistributive"]
    train_epochs = 100
    emb_size = 128
    learning_rate = 0.001

    # we will save all results in this directory
    results_dir = f"results/igcore/"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset']
    for dataset, label_name, temperature in zip(datasets, label_names, temperatures):

        # load data and set up cross-validation
        data = load_data(dataset, label_name=label_name, root_dir='../gnn4ua/datasets/')
        n_classes = data.y.shape[1]
        y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in data.y])
        skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

        for fold, (train_index, test_index) in enumerate(skf.split(data.y, y_new)):
            # define model
            gnn = GCoRe(data.x.shape[1], emb_size, n_classes)

            optimizer = torch.optim.AdamW(gnn.parameters(), lr=learning_rate)
            loss_form = torch.nn.CrossEntropyLoss(weight=1-data.y.float().mean(dim=0))
            # loss_form = torch.nn.CrossEntropyLoss()
            gnn.train()
            for epoch in range(train_epochs):
                optimizer.zero_grad()
                y_pred = gnn.forward(data)
                loss = loss_form(y_pred[train_index], data.y[train_index].argmax(dim=1))
                loss.backward()
                optimizer.step()

                # monitor AUC
                if epoch % 1 == 0:
                    train_auc = roc_auc_score(data.y[train_index].detach(), y_pred[train_index].detach())
                    train_accuracy = accuracy_score(data.y[train_index].detach(), y_pred[train_index].detach()>0.5)
                    print(f'Epoch {epoch}: loss {loss:.4f} train AUC: {train_auc:.4f} train accuracy: {train_accuracy:.4f}')

            # make predictions on test set and evaluate results
            test_auc = roc_auc_score(data.y[test_index].detach(), y_pred[test_index].detach())
            print(f'Test accuracy: {test_auc:.4f}')
            results.append(['', test_auc, fold, 'GCoRe (ours)', label_name])
            pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))


if __name__ == '__main__':
    main()
