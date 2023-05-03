import joblib
import pandas as pd
import torch
import os
import numpy as np
import torch_geometric as pyg


def load_data(dataset_name, label_name, root_dir='../gnn4ua/datasets/', generalization='strong',
              random_state=42, max_size_train=40, max_prob_train=0.8):
    file_name = os.path.join(root_dir, f'{dataset_name}_{label_name}.joblib')

    if not os.path.exists(file_name):
        file_name = os.path.join(root_dir, dataset_name+'.json')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f'File {file_name} does not exist')

        df = pd.read_json(file_name, lines=True)
        adjacency_matrices = df['Adj_matrix'].values.tolist()

        # Compute the indices of the nonzero elements in the adjacency matrices
        edge_indices = []
        labels = []
        batch = []
        train_mask = []
        test_mask = []
        n_nodes_seen = 0
        for i, matrix in enumerate(adjacency_matrices):
            # TODO: make matrix symmetric?
            # generate the indices of the nonzero elements in the adjacency matrix
            matrix_with_self_loops = np.array(matrix) + np.eye(len(matrix))
            matrix_indices = torch.nonzero(torch.tensor(matrix_with_self_loops, dtype=torch.float)).tolist()
            matrix_indices_shifted = [(index[0] + n_nodes_seen, index[1] + n_nodes_seen) for index in matrix_indices]
            edge_indices.extend(matrix_indices_shifted)

            # generate the labels
            label = df[label_name].values[i].astype(int)
            labels.append(torch.LongTensor([1-label, label]))

            # generate the batch index
            batch.extend(torch.LongTensor([i] * len(matrix)))

            # generate the train/test masks
            if generalization == 'strong':
                if len(matrix) > max_size_train:
                    is_train_graph = False
                elif len(matrix) < max_size_train:
                    is_train_graph = True
                elif len(matrix) == max_size_train:
                    if np.random.RandomState(random_state*(i+1)).rand() < max_prob_train:
                        is_train_graph = True
                    else:
                        is_train_graph = False
            else:
                is_train_graph = True if np.random.RandomState(random_state*(i+1)).rand() < max_prob_train else False
            train_mask.extend(torch.BoolTensor([is_train_graph]))
            test_mask.extend(torch.BoolTensor([not is_train_graph]))

            n_nodes_seen += len(matrix)

        # Create the edge index tensor
        edge_index = torch.tensor(edge_indices).t()
        x = torch.ones((edge_index.max().item() + 1, 1))
        y = torch.vstack(labels)
        batch = torch.hstack(batch)
        train_mask = torch.hstack(train_mask)
        test_mask = torch.hstack(test_mask)

        data = pyg.data.Data(x=x, edge_index=edge_index, y=y, batch=batch, train_mask=train_mask, test_mask=test_mask)
        data.validate(raise_on_error=True)

        joblib.dump(data, os.path.join(root_dir, f'{dataset_name}_{label_name}_{generalization}.joblib'))
    else:
        data = joblib.load(os.path.join(root_dir, f'{dataset_name}_{label_name}_{generalization}.joblib'))
        data.validate(raise_on_error=True)
    return data
