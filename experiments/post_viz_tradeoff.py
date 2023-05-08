import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import os
from torch_geometric import seed_everything
import seaborn as sns

import sys

sys.path.append('..')
from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN, GCoRe

# torch.autograd.set_detect_anomaly(True)
seed_everything(42)

# viz parameters
n_examples_per_concept = 5
n_concepts = 20
n_hops_neighborhood = 8
plt.rcParams.update({
    "text.usetex": True,
})
sns.set_style("whitegrid")

def main():
    # we will save all results in this directory
    results_dir = f"results/metrics/"
    metrics = ['completeness_fidelity']

    figures_dir = f"results/figures/"
    plot_file = os.path.join(figures_dir, f'completeness_fidelity')

    for metric in metrics:
        results = pd.read_csv(os.path.join(results_dir, f'{metric}.csv'))

        generalization_modes = results['generalization'].unique()
        tasks = results['task'].unique()
        models = results['model'].unique()

        cols = sns.color_palette("Paired")[:6]
        cmap = ListedColormap(sns.color_palette(cols).as_hex())

        plt.figure(figsize=(23, 3))
        for task_id, task in enumerate(tasks):
            task_name = task.replace('_', '')
            results_task = results[results['task'] == task]

            ax = plt.subplot(1, len(tasks), task_id+1)
            plt.title(F'{task_name}', fontsize=16)
            col_id = 0
            legend_patches = []
            for model in models:
                for generalization_mode in generalization_modes:
                    results_task_gb = results_task.groupby(['model', 'generalization'])
                    results_task_mean = results_task_gb.mean()*100
                    results_task_sem = results_task_gb.sem()*100
                    ax.errorbar(results_task_mean.loc[model].loc[generalization_mode, 'completeness'],
                                 results_task_mean.loc[model].loc[generalization_mode, 'fidelity'],
                                 results_task_sem.loc[model].loc[generalization_mode, 'completeness'],
                                 results_task_sem.loc[model].loc[generalization_mode, 'fidelity'],
                                 capsize=5, elinewidth=3, capthick=2, fmt='.', color=cols[col_id])
                    ax.set_xlabel('COMPLETENESS \%', fontsize=13)
                    if task_id == 0:
                        ax.set_ylabel('FIDELITY \%', fontsize=13)

                    if model == 'GCoRe':
                        model_name = 'iGNN'
                    elif model == 'BlackBoxGNN':
                        model_name = 'Black-Box GNN'
                    else:
                        model_name = 'HiGNN'
                    legend_patches.append(Patch(facecolor=cols[col_id], label=f'{model_name} [{generalization_mode} mode]'))
                    col_id += 1

            ax.set_xlim(right=100)
            ax.set_ylim(top=100.5)

        plt.legend(handles=legend_patches, fontsize=12, loc='upper center',
          # bbox_to_anchor=(-0.8, -0.2),
          bbox_to_anchor=(-2.8, -0.2), frameon=False,
           fancybox=False, shadow=False, ncol=6)
        # plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.05))
        plt.savefig(f'{plot_file}.png', bbox_inches='tight')
        plt.savefig(f'{plot_file}.pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
