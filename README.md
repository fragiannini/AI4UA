# AI4UA

# Torch geometric installation
Follow instructions in https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

> pip install torch_geometric
> 
> pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Training
To run the code,

> cd experiments
>
> python run.py

# TODO

- [x] Return Gumbel-Softmax outputs
- [x] Save model weights
- [x] Plot concepts (subgraphs) such that the higher the ID, the higher the node appears in the figure (directed graph)
- [x] Setup out-of-distribution experiments (after the whole pipeline is built)
- [ ] Plot t-SNE reduction of the embedding space
- [ ] Write nice code to compute concept completeness and purity
- [ ] Start writing the paper (**after all experiments are done**)


# Contributing
Please use these [Gitmojis](https://gist.github.com/akoepcke/36598d90b0864ebd752b360f5ccb379d) 
to flag the content of your commit messages 😉.
