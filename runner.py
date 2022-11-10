from utils import generate_all_lattices_classes, prepare_dataset_json_class

num_lattices_to_sample = 8  # 10
max_cardinality_to_generate_all = 8
num = 10  # max number of nodes for lattices to generate

lattices = generate_all_lattices_classes(num, max_cardinality_to_generate_all, num_lattices_to_sample)
prepare_dataset_json_class(lattices)
