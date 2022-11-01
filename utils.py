import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#TODO
def is_modular(mat):
    n = np.size(mat[0])

    mod = True
    return mod

def is_distributive(mat):
    n = np.size(mat[0])

    distr = True

    for x in range(n):
        for y in range(n):
            for z in range(n):
                if meet(x,join(y,z,mat),mat) != join(meet(x,y,mat),meet(x,z,mat),mat):
                    distr = False
                    return distr
    return distr

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
                            return trans
                    k +=1
            j += 1
        i += 1

    return trans

def meet(a,b,less_or_equal):
    n = np.size(less_or_equal[0])
    minority = [c for c in range(n) if (less_or_equal[c,a] == 1 and less_or_equal[c,b] == 1)]
    inf = max(minority)
    return inf

def join(a,b,less_or_equal):
    n = np.size(less_or_equal[0])
    majority = [c for c in range(n) if (less_or_equal[a,c] == 1 and less_or_equal[b,c] == 1)]
    sup = min(majority)
    return sup



def plot_graph_from_adiacency(adjacency_matrix):
    G = nx.DiGraph()
    n = np.size(adjacency_matrix[0])
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i,j)

    TR = nx.transitive_reduction(G)

    # print(TR.adj)

    # pos = nx.nx_pydot.pydot_layout(,adjacency_matrix,root=0)
    # degree = [n for n in G.in_degree()]
    # print(degree)

    # nx.draw(TR, labels={i:str(i) for i in range(n)}, pos=pos)
    nx.draw(TR, labels={i:str(i) for i in range(n)})
    plt.show()
