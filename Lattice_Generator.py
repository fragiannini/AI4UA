import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt


def is_transitive(mat):
    n = np.size(mat[0])

    trans = True

    i = 1
    while i < n-1:
        j = i+1
        while j < n-1:
            if mat[i,j] == 1:
                k = j + 1
                while k < n - 1:
                    if mat[j,k] == 1:
                        if mat[i,k] == 0:
                            trans = False
                    k +=1
            j += 1
        i += 1

    return trans


def generate_adiacency_matrices(n,domain_pairs,assignments, sampling=None):
    matrices_list = []

    if sampling:
        random_assignments = np.random.rand(assignments)[:10]

        for a in random_assignments:
            new_matrix = np.triu(np.ones([n, n]), k=1)

            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = a[j]

            if is_transitive((new_matrix)):
                matrices_list.append(new_matrix)

    else:
        for a in assignments:
            new_matrix = np.triu(np.ones([n,n]),k=1)

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

    matrices_list = generate_adiacency_matrices(n,domain_pairs,assignments)
    return matrices_list




def plot_graph_from_adiacency(adjacency_matrix):
    G = nx.DiGraph()
    n = np.size(adjacency_matrix[0])
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i,j)

    TR = nx.transitive_reduction(G)

    print(TR.adj)

    pos = nx.planar_layout(G)
    # degree = [n for n in G.in_degree()]
    # print(degree)

    nx.draw(TR, labels={i:str(i) for i in range(n)}, pos=pos)
    plt.show()





lattices = generate_all_lattices(4)
print(lattices)


for i in lattices:
    plot_graph_from_adiacency(i)


