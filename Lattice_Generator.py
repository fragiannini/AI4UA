from utils import generate_all_lattices, prepare_dataset_json

num_lattices_to_sample = 8  # 10
max_cardinality_to_generate_all = 8
num = 7  # max number of nodes for lattices to generate

lattices = generate_all_lattices(num,max_cardinality_to_generate_all,num_lattices_to_sample)
prepare_dataset_json(lattices)


# for i in lattices:
#     # print(is_distributive(i))
#     plot_graph_from_adiacency(LoE2Adj(i))