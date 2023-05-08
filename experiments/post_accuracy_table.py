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
    # we will save all results in this directory
    results_dir = f"results/metrics/"
    metrics = ['auc', 'completeness']

    for metric in metrics:
        results = pd.read_csv(os.path.join(results_dir, f'{metric}.csv'))

        generalization_modes = results['generalization'].unique()

        for generalization_mode in generalization_modes:
            results_accuracy_groups = results[results['generalization'] == generalization_mode].groupby(['model', 'task'])
            results_accuracy_mean = results_accuracy_groups['test_auc'].mean().reset_index()
            results_accuracy_sem = results_accuracy_groups['test_auc'].sem().reset_index()
            tasks = results_accuracy_mean['task'].unique()
            models = results_accuracy_mean['model'].unique()

            results_list_str = []
            for m in models:
                rows_mean = results_accuracy_mean[results_accuracy_mean['model'] == m]
                rows_sem = results_accuracy_sem[results_accuracy_sem['model'] == m]
                new_row = [m]
                for row_mean, row_sem in zip(rows_mean['test_auc'].values, rows_sem['test_auc'].values):
                    new_row.extend([f'${100 * row_mean:.2f} \pm {100 * row_sem:.2f}$'])
                results_list_str.append(new_row)

            results_table = pd.DataFrame(results_list_str, columns=['model'] + list(tasks))
            results_table.T.to_csv(os.path.join(results_dir, f'paper_{metric}_{generalization_mode}.csv'), header=None)


if __name__ == '__main__':
    main()
