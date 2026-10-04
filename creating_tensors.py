import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import PIL
import os
import math
import cv2
from visualiser import generate_images


def make_training_tensor(path_to_dataset,test_train_split, number_of_images_per_molecule,name_of_decoys_folder,name_of_ligands_folder,type_of_file,has_first_row_as_atom_names):
    resolution = 100
    rgb_channels = 3
    path_to_decoys = os.path.join(path_to_dataset, name_of_decoys_folder)
    path_to_ligand = os.path.join(path_to_dataset, name_of_ligands_folder)

    print((int(len(os.listdir(path_to_decoys))*test_train_split),number_of_images_per_molecule,rgb_channels,resolution,resolution))
    training_decoys = torch.zeros(int(len(os.listdir(path_to_decoys))*test_train_split),number_of_images_per_molecule,rgb_channels,resolution,resolution)
    validation_decoys = torch.zeros(len(os.listdir(path_to_decoys)) - training_decoys.shape[0],number_of_images_per_molecule,rgb_channels,resolution,resolution)
    training_ligands = torch.zeros(int(len(os.listdir(path_to_ligand))*test_train_split),number_of_images_per_molecule,rgb_channels,resolution,resolution)
    validation_ligands = torch.zeros(len(os.listdir(path_to_ligand)) - training_ligands.shape[0],number_of_images_per_molecule,rgb_channels,resolution,resolution)
    all_paths = [path_to_decoys, path_to_ligand]

    for folder_path in all_paths:
        for folder_index, folder_path in enumerate(all_paths):
            items = os.listdir(folder_path)
            split = int(len(items) * test_train_split)

            for item_index, item in enumerate(items):
                if type_of_file == "csv":
                    data = pd.read_csv(os.path.join(folder_path, item))
                elif type_of_file == "excel":
                    data = pd.read_excel(os.path.join(folder_path, item))
                if item_index < split:
                    # Training
                    if folder_index == 0:
                        print(item, "decoy", "training")
                        training_decoys[item_index] = generate_images(data,number_of_images=number_of_images_per_molecule,speed=1,rad=(0,1),atom_type=1,res=resolution,zoom=0.05,has_first_row_as_atom_names=has_first_row_as_atom_names)
                    else:
                        print(item, "ligand", "training")
                        training_ligands[item_index] = generate_images(data,number_of_images=number_of_images_per_molecule,speed=1,rad=(0,1),atom_type=1,res=resolution,zoom=0.05,has_first_row_as_atom_names=has_first_row_as_atom_names)
                else:
                    # Validation
                    if folder_index == 0:
                        print(item, "decoy", "validation")
                        validation_decoys[item_index - training_decoys.shape[0]] = generate_images(data,number_of_images=number_of_images_per_molecule,speed=1,rad=(0,1),atom_type=1,res=resolution,zoom=0.05,has_first_row_as_atom_names=has_first_row_as_atom_names)
                    else:
                        print(item, "ligand", "validation")
                        validation_ligands[item_index - training_ligands.shape[0]] = generate_images(data,number_of_images=number_of_images_per_molecule,speed=1,rad=(0,1),atom_type=1,res=resolution,zoom=0.05,has_first_row_as_atom_names=has_first_row_as_atom_names)

    labels_training = labels = torch.cat([torch.zeros(int(len(os.listdir(path_to_decoys))*test_train_split),number_of_images_per_molecule), torch.ones(int(len(os.listdir(path_to_ligand))*test_train_split),number_of_images_per_molecule)],dim=0)
    labels_validation = labels = torch.cat([torch.zeros(len(os.listdir(path_to_decoys)) - training_decoys.shape[0],number_of_images_per_molecule), torch.ones(len(os.listdir(path_to_ligand)) - training_ligands.shape[0],number_of_images_per_molecule)],dim=0)
    
    
    torch.save(training_decoys, 'tensors/training_decoys.pt')
    torch.save(validation_decoys, 'tensors/validation_decoys.pt')
    torch.save(training_ligands, 'tensors/training_ligands.pt')
    torch.save(validation_ligands, 'tensors/validation_ligands.pt')

    torch.save(labels_training, 'tensors/labels_training.pt')
    torch.save(labels_validation, 'tensors/labels_validation.pt')
    print(training_decoys.shape,validation_decoys.shape,training_ligands.shape,validation_ligands.shape)



