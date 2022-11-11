from utils import generate_all_lattices, prepare_dataset_json, plot_graph_from_lattice


num_lattices_to_sample = 10  # 10
max_cardinality_to_generate_all = 7
num = 100  # max number of nodes for lattices to generate

lattices = generate_all_lattices(num, max_cardinality_to_generate_all, num_lattices_to_sample)
prepare_dataset_json(lattices)

# for lattice in lattices:
#     plot_graph_from_lattice(lattice)




