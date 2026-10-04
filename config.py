from creating_tensors import make_training_tensor
import creating_tensors
path_to_dataset = '737 copy'
master_resolution = 100
make_training_tensor(path_to_dataset,
                      test_train_split = 0.7,
                      number_of_images_per_molecule=6,
                      name_of_decoys_folder='decoys',
                      name_of_ligands_folder='ligs',
                      type_of_file = "csv",
                      has_first_row_as_atom_names = False)