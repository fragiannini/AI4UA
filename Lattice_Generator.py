import numpy as np
import itertools
from utils import is_distributive, is_transitive, is_modular
import json

lattices_to_generate = 10 # 10
max_lattices_cardinality = 8


#TODO
def LoQ2Adj(mat):
    n = np.size(mat[0])

    adjacency_matrix = np.zeros([n,n])
    return adjacency_matrix

def generate_LoQ_matrices(n,domain_pairs, sampling, lattices_to_generate):
    matrices_list = []

    # number of pairs to define possible functions
    num_of_pairs = len(domain_pairs)

    if sampling:
        tuple_taken = len(matrices_list)
        while tuple_taken < lattices_to_generate:
            candidate = np.random.choice([0,1], size=num_of_pairs)

            new_matrix = np.triu(np.ones([n, n])) # , k=1
            # new_matrix = np.triu(np.ones([n, n]), k=1)
            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = candidate[j]
            if is_transitive(new_matrix):
                # if new_matrix not in matrices_list:
                matrices_list.append(new_matrix)
                tuple_taken += 1

    else:
        assignments = itertools.product([0, 1], repeat=num_of_pairs)
        for a in assignments:
            new_matrix = np.triu(np.ones([n,n])) # , k=1
            # new_matrix = np.triu(np.ones([n,n]), k=1)

            for j, p in enumerate(domain_pairs):
                new_matrix[p[0],p[1]] = a[j]

            if is_transitive((new_matrix)):
                matrices_list.append(new_matrix)

    matrices_list = list(matrices_list)

    return matrices_list



def generate_lattices(n):
    domain_pairs = [[i, j] for i in range(1, n-1) for j in range(i + 1, n-1)]

    if n > max_lattices_cardinality:
        sampling = True
    else:
        sampling = False

    matrices_list_for_cardinality = generate_LoQ_matrices(n,domain_pairs,sampling,lattices_to_generate)
    return matrices_list_for_cardinality

def generate_all_lattices(n):
    lattices_list = []
    for i in range(2,n+1):
        lattices_list += generate_lattices(i)
    return lattices_list




def prepare_dataset_json(lattices):
    # FIELD: graph_cardinality; LoQ_mat; adj_mat; distribut; modularity

    with open("sample.json", "w") as outfile:
        for i, latt in enumerate(lattices):
            dictionary = {
                "ID": "G"+str(i),
                "Cardinality": np.size(latt[0]),
                "LoQ_matrix": latt.tolist(),
                "Adj_matrix": LoQ2Adj(latt).tolist(),
                "Distributive": is_distributive(latt),
                "Modular": is_modular(latt)
            }
            # create and write json lattice
            json_object = json.dumps(dictionary)
            outfile.write(json_object)

    outfile.close()



lattices = generate_all_lattices(5)

prepare_dataset_json(lattices)



for i in lattices:
    print(is_distributive(i))
    # plot_graph_from_adiacency(i)


