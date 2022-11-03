import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def is_modular(mat):
    n = np.size(mat[0])

    modular = True

    for x in range(n):
        for y in range(n):
            for z in range(n):
                if join(meet(x,y,mat),meet(z,y,mat),mat) != meet(join(meet(x,y,mat),z,mat),y,mat):
                    modular = False
                    return modular
    return modular

def is_distributive(mat):
    n = np.size(mat[0])

    distributive = True

    for x in range(n):
        for y in range(n):
            for z in range(n):
                if meet(x,join(y,z,mat),mat) != join(meet(x,y,mat),meet(x,z,mat),mat):
                    distributive = False
                    return distributive
    return distributive

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

def meet(a,b,LoE):
    n = np.size(LoE[0])
    minority = [c for c in range(n) if (LoE[c,a] == 1 and LoE[c,b] == 1)]
    inf = max(minority)
    return inf

def join(a,b,LoE):
    n = np.size(LoE[0])
    majority = [c for c in range(n) if (LoE[a,c] == 1 and LoE[b,c] == 1)]
    sup = min(majority)
    return sup

def is_lattice(LoE):
    n = np.size(LoE[0])

    for a in range(1,n-1):
        for b in range(1, n - 1):
            minority = [c for c in range(n) if (LoE[c, a] == 1 and LoE[c, b] == 1)]
            inf = max(minority)
            for m in minority:
                if LoE[m, inf] == 0:
                    return False

            majority = [c for c in range(n) if (LoE[a, c] == 1 and LoE[b, c] == 1)]
            sup = min(majority)
            for M in majority:
                if LoE[sup, M] == 0:
                    return False

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
