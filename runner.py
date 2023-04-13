from utils import generate_all_lattices, prepare_dataset_json, plot_graph_from_lattice

max_cardinality_to_generate_all_lattices = 7 # until this cardinality we'll generate all the lattices
num_lattices_to_sample = 10  # number of lattices generated per size beyond the max cardinality allowed
max_cardinality_to_generate_lattices = 20  # max number of nodes for the lattices we generate
plot_lattices = False

lattices = generate_all_lattices(max_cardinality_to_generate_lattices, max_cardinality_to_generate_all_lattices, num_lattices_to_sample)
prepare_dataset_json(lattices)



if plot_lattices:
    for lattice in lattices:
        plot_graph_from_lattice(lattice)




