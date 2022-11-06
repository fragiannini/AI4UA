import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import itertools
from lattice import Lattice


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
    minority = [c for c in range(min(a, b)+1) if (LoE[c, a] == 1 and LoE[c, b] == 1)]
    if minority == []:
        print(LoE)
    inf = max(minority)
    return inf


def join(a, b, LoE):
    n = np.size(LoE[0])
    majority = [c for c in range(max(a, b), n) if (LoE[a, c] == 1 and LoE[b, c] == 1)]
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
    #Todo: c'è un errore da qualche parte
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

def generate_LoE_matrices(n, domain_pairs, sampling, num_lattices_to_sample):
    matrices_list = []

    # number of pairs to define possible functions
    num_of_pairs = len(domain_pairs)

    if sampling:
        tuple_taken = len(matrices_list)
        while tuple_taken < num_lattices_to_sample:
            candidate = np.random.choice([0, 1], size=num_of_pairs)

            new_matrix = np.triu(np.ones([n, n]))  # , k=1
            # new_matrix = np.triu(np.ones([n, n]), k=1)
            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = candidate[j]
            new_matrix = transitive_closure(new_matrix)
            # if is_lattice(new_matrix): # and not has_isomorphic(new_matrix,matrices_list):
            #if is_lattice(new_matrix) and not has_isomorphic(new_matrix,matrices_list):
            if not has_isomorphic(new_matrix, matrices_list):
                matrices_list.append(new_matrix)
                tuple_taken += 1

    else:
        assignments = itertools.product([0, 1], repeat=num_of_pairs)
        for a in assignments:
            new_matrix = np.triu(np.ones([n, n]))  # , k=1
            # new_matrix = np.triu(np.ones([n,n]), k=1)

            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = a[j]
            new_matrix = transitive_closure(new_matrix)
            # if is_lattice(new_matrix): # and not has_isomorphic(new_matrix,matrices_list):
            #if is_lattice(new_matrix) and not has_isomorphic(new_matrix, matrices_list):
            if not has_isomorphic(new_matrix, matrices_list):
                matrices_list.append(new_matrix)

    return matrices_list


def generate_lattices(n,max_cardinality_to_generate_all,num_lattices_to_sample):
    domain_pairs = [[i, j] for i in range(1, n - 1) for j in range(i + 1, n - 1)]

    if n > max_cardinality_to_generate_all:
        sampling = True
    else:
        sampling = False

    matrices_list_for_cardinality = generate_LoE_matrices(n, domain_pairs, sampling, num_lattices_to_sample)
    return matrices_list_for_cardinality


def generate_lattices_classes(n, max_cardinality_to_generate_all, num_lattices_to_sample):
    domain_pairs = [[i, j] for i in range(1, n - 1) for j in range(i + 1, n - 1)]
    if n > max_cardinality_to_generate_all:
        sampling = True
    else:
        sampling = False

    matrices_list_for_cardinality = generate_LoE_matrices(n, domain_pairs, sampling, num_lattices_to_sample)
    lattices = [Lattice(loe) for loe in matrices_list_for_cardinality]
    return lattices


def generate_all_lattices(n,max_cardinality_to_generate_all,num_lattices_to_sample):
    print("SETTING: ","Generate lattices up to",n,"elements;", " Max cardinality to generate all:",max_cardinality_to_generate_all,"; Num of samples:",num_lattices_to_sample)

    lattices_list = []
    for i in range(2, n + 1):
        print("generating lattices with ",i," elements")
        lattices_list += generate_lattices(i,max_cardinality_to_generate_all,num_lattices_to_sample)
    return lattices_list


def generate_all_lattices_classes(n, max_cardinality_to_generate_all, num_lattices_to_sample):
    print("SETTING: ","Generate lattices up to", n, "elements;", " Max cardinality to generate all:",
          max_cardinality_to_generate_all, "; Num of samples:", num_lattices_to_sample)

    lattices_list = []
    for i in range(2, n + 1):
        print("generating lattices with ", i, " elements")
        lattices_list += generate_lattices_classes(i, max_cardinality_to_generate_all, num_lattices_to_sample)
    return lattices_list


def prepare_dataset_json(lattices):
    # FIELD: graph_cardinality; LoE_mat; adj_mat; distribut; modularity

    with open("sample.json", "w") as outfile:
        for i, latt in enumerate(lattices):
            #preparing fields
            ID = "G" + str(i)
            card = np.size(latt[0])
            LoE_mat = latt
            Adj_mat = LoE2Adj(latt)
            Distr = is_distributive(latt)
            Mod = is_modular(latt)

            dictionary = {
                "ID": ID,
                "Cardinality": card,
                "LoE_matrix": LoE_mat.tolist(),
                "Adj_matrix": Adj_mat.tolist(),
                "Distributive": Distr,
                "Modular": Mod
            }
            # create and write json lattice
            json_object = json.dumps(dictionary)
            outfile.write(json_object + "\n")

    outfile.close()

def prepare_dataset_json_class(lattices):
    # FIELD: graph_cardinality; LoE_mat; adj_mat; distribut; modularity

    with open("sample.json", "w") as outfile:
        for i, latt in enumerate(lattices):
            if latt.is_a_lattice:
                #preparing fields
                ID = "G" + str(i)
                card = latt.size
                LoE_mat = latt.loe
                Adj_mat = latt.adj
                Distr = latt.dist
                Mod = latt.mod

                dictionary = {
                    "ID": ID,
                    "Cardinality": card,
                    "LoE_matrix": LoE_mat.tolist(),
                    "Adj_matrix": Adj_mat.tolist(),
                    "Distributive": Distr,
                    "Modular": Mod
                }
                # create and write json lattice
                json_object = json.dumps(dictionary)
                outfile.write(json_object + "\n")

    outfile.close()
