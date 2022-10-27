import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from networkx import nx_pydot
from utils import *

number_of_lattices_to_generate = 9




def generate_LoQ_matrices(n,domain_pairs,assignments, sampling):
    matrices_list = []

    if sampling:
        random_assignments = np.random.rand(assignments)[:number_of_lattices_to_generate]

        for a in random_assignments:
            new_matrix = np.triu(np.ones([n, n]))

            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = a[j]

            if is_transitive((new_matrix)):
                matrices_list.append(new_matrix)

    else:
        for a in assignments:
            new_matrix = np.triu(np.ones([n,n]))

            for j, p in enumerate(domain_pairs):
                new_matrix[p[0],p[1]] = a[j]

            if is_transitive((new_matrix)):
                matrices_list.append(new_matrix)

    return(matrices_list)



def generate_all_lattices(n):

    domain_pairs = []
    x = 1
    while x < n-1:
        y = x + 1
        while y < n-1:
            domain_pairs.append([x,y])
            y += 1
        x += 1

    num_of_pairs = len(domain_pairs)
    assignments = itertools.product([0,1],repeat=num_of_pairs)

    if n >= 10:
        sampling = True
    else:
        sampling = False

    matrices_list = generate_LoQ_matrices(n,domain_pairs,assignments,sampling)
    return matrices_list




def plot_graph_from_adiacency(adjacency_matrix):
    G = nx.DiGraph()
    n = np.size(adjacency_matrix[0])
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i,j)

    # G = nx.transitive_reduction(G)

    # print(TR.adj)

    # pos = nx.nx_pydot.pydot_layout(,adjacency_matrix,root=0)
    # degree = [n for n in G.in_degree()]
    # print(degree)

    # nx.draw(TR, labels={i:str(i) for i in range(n)}, pos=pos)
    nx.draw(G, labels={i:str(i) for i in range(n)})
    plt.show()





lattices = generate_all_lattices(9)


for i in lattices:
    print(is_distributive(i))
    # plot_graph_from_adiacency(i)


