import numpy as np
import json
from utils import LoE2Adj


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


def generate_all_lattices(n,max_cardinality_to_generate_all,num_lattices_to_sample):
    print("SETTING: ","Generate lattices up to",n,"elements;", " Max cardinality to generate all:",max_cardinality_to_generate_all,"; Num of samples:",num_lattices_to_sample)

    lattices_list = []
    for i in range(2, n + 1):
        print("generating lattices with ",i," elements")
        lattices_list += generate_lattices(i,max_cardinality_to_generate_all,num_lattices_to_sample)
    return lattices_list

def generate_lattices(n,max_cardinality_to_generate_all,num_lattices_to_sample):
    domain_pairs = [[i, j] for i in range(1, n - 1) for j in range(i + 1, n - 1)]

    if n > max_cardinality_to_generate_all:
        sampling = True
    else:
        sampling = False

    matrices_list_for_cardinality = generate_LoE_matrices(n, domain_pairs, sampling, num_lattices_to_sample)
    return matrices_list_for_cardinality


def LoE2Adj(mat, reflexive=False):
    n = np.size(mat[0])
    adj = np.copy(mat)

    if not reflexive:
        for i in range(n):
            adj[i, i] = 0

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                for k in range(j + 1, n):
                    if adj[j, k] == 1:
                        adj[i, k] = 0
    return adj