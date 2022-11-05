import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def is_modular(mat):
    n = np.size(mat[0])

    for x in range(n):
        for y in range(n):
            for z in range(n):
                if join(meet(x, y, mat), meet(z, y, mat), mat) != meet(join(meet(x, y, mat), z, mat), y, mat):
                    return False
    return True


def is_distributive(mat):
    n = np.size(mat[0])

    for x in range(n):
        for y in range(n):
            for z in range(n):
                if meet(x, join(y, z, mat), mat) != join(meet(x, y, mat), meet(x, z, mat), mat):
                    return False
    return True


def is_transitive(mat):
    n = np.size(mat[0])
    # assert mat[0,:]==mat[n-1,:]==1

    i = 1
    while i < n - 1:
        j = i + 1
        while j < n - 1:
            if mat[i, j] == 1:
                k = j + 1
                while k < n - 1:
                    if mat[j, k] == 1:
                        if mat[i, k] == 0:
                            return False
                    k += 1
            j += 1
        i += 1
    return True


def transitive_closure(mat):
    n = np.size(mat[0])

    for i in range(n):
        mat_trans = np.where(np.matmul(mat, mat) > 0, 1., 0.)
        if np.array_equal(mat_trans, mat):
            return mat_trans
        mat = mat_trans
    return mat


def meet(a, b, LoE):
    n = np.size(LoE[0])
    minority = [c for c in range(n) if (LoE[c, a] == 1 and LoE[c, b] == 1)]
    inf = max(minority)
    return inf


def join(a, b, LoE):
    n = np.size(LoE[0])
    majority = [c for c in range(n) if (LoE[a, c] == 1 and LoE[b, c] == 1)]
    sup = min(majority)
    return sup


# def is_lattice(LoE):
#     n = np.size(LoE[0])
#
#     for a in range(1, n - 1):
#         for b in range(1, n - 1):
#             minority = [c for c in range(n) if (LoE[c, a] == 1 and LoE[c, b] == 1)]
#             inf = max(minority)
#             for m in minority:
#                 if LoE[m, inf] == 0:
#                     return False
#
#             majority = [c for c in range(n) if (LoE[a, c] == 1 and LoE[b, c] == 1)]
#             sup = min(majority)
#             for M in majority:
#                 if LoE[sup, M] == 0:
#                     return False
#     return True


def is_lattice(LoE):
    n = np.size(LoE[0])
    adj = LoE2Adj(LoE,reflexive=True)

    for a in range(1, n - 1):
        for b in range(a+1, n - 1):
            strict_minority = [c for c in range(min(a,b)+1) if (adj[c, a] == 1 and adj[c, b] == 1)]
            if len(strict_minority) > 1:
                return False

            strict_majority = [c for c in range(max(a,b),n) if (adj[a, c] == 1 and adj[b, c] == 1)]
            if len(strict_majority) > 1:
                return False

    return True


def LoE2Adj(mat,reflexive=False):
    n = np.size(mat[0])
    adj = np.copy(mat)

    if not reflexive:
        for i in range(n):
            adj[i,i]=0

    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                for k in range(j+1, n):
                    if adj[j, k] == 1:
                        adj[i, k] = 0
    return adj


# #TODO: check if a lattice (function on dom_pairs) is isomorph to atleast one in a list
# #TODO: check if it works or if some pairs of lattices not iso are taken
# def is_isomorph(mat,mat_list,dom_pairs):
#     ass_mat = [mat[i,j] for [i,j] in dom_pairs]
#     ass_mat_sorted = ass_mat.sort()
#     for m in mat_list:
#         ass_m = [mat[i, j] for [i, j] in dom_pairs]
#         ass_m_sorted = ass_m.sort()
#         if ass_mat_sorted == ass_m_sorted:
#             return True
#     return False

def has_isomorphic(latt,latt_list):
    G = nx.from_numpy_matrix(LoE2Adj(latt),create_using=nx.DiGraph)
    for mat in latt_list:
        G_mat = nx.from_numpy_matrix(LoE2Adj(mat),create_using=nx.DiGraph)
        if nx.is_isomorphic(G, G_mat):
            return True
    return False



def plot_graph_from_adiacency(adjacency_matrix):
    G = nx.DiGraph()
    n = np.size(adjacency_matrix[0])
    for i in range(n):
        for j in range(i+1,n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)

    # TR = nx.transitive_reduction(G)
    # TR = G

    # nx.draw(TR, labels={i:str(i) for i in range(n)}, pos=pos)
    nx.draw(G, labels={i: str(i) for i in range(n)}, pos=nx.planar_layout(G))
    plt.show()
