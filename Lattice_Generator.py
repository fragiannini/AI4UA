import numpy as np
import itertools
from utils import is_distributive, has_isomorphic, is_modular, plot_graph_from_adiacency, transitive_closure, is_lattice, \
    LoE2Adj
import json

num_lattices_to_sample = 8  # 10
max_cardinality_to_generate_all = 8
num = 30  # max number of nodes for lattices to generate


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
            if is_lattice(new_matrix) and not has_isomorphic(new_matrix,matrices_list):
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
            if is_lattice(new_matrix) and not has_isomorphic(new_matrix,matrices_list):
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


def generate_all_lattices(n,max_cardinality_to_generate_all,num_lattices_to_sample):
    print("SETTING: ","Generate lattices up to",n,"elements;", " Max cardinality to generate all:",max_cardinality_to_generate_all,"; Num of samples:",num_lattices_to_sample)

    lattices_list = []
    for i in range(2, n + 1):
        print("generating lattices with ",i," elements")
        lattices_list += generate_lattices(i,max_cardinality_to_generate_all,num_lattices_to_sample)
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


lattices = generate_all_lattices(num,max_cardinality_to_generate_all,num_lattices_to_sample)
prepare_dataset_json(lattices)


# for i in lattices:
#     # print(is_distributive(i))
#     plot_graph_from_adiacency(LoE2Adj(i))
