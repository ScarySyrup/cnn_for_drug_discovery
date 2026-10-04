import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import PIL
import os
import math
import cv2



#encode the atoms present in the datset as points on a circle in R2
def atoms_encoding(atom):
    present_atoms = ["C","H","O","N","S","F","Cl","Br","I","P"]
    no_of_present_atoms = len(present_atoms)
    index = present_atoms.index(atom)
    return [math.cos(2*torch.pi*index/no_of_present_atoms),math.sin(2*torch.pi*index/no_of_present_atoms)]



max_point_mag = 10




#reading and processing the old dataset









#deleting all images in a directory
def delete_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff')):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)



#rotate a point
def rotate(x,y,z):
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x),  torch.cos(x)]
    ])
    R_y = torch.tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ])
    R_z = torch.tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z),  torch.cos(z), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x


#project any 3d point to a 2d one
def project_to_isometric(t):
    return torch.stack([((3)**0.5)/2*(t[:,0]-t[:,1]), 0.5*(t[:,0]+t[:,1])-t[:,2]], dim=1)




#creating the grid of pixels
def process_data(data,zoom_val=0.05,naming_index = 0,rotation_angles = torch.tensor([0.0, 0.0, 0.0]),rad=(0,1),format = "full",atom_type=1,res=400,has_first_row_as_atom_names=False):
    
    datafr = data.copy()
    if has_first_row_as_atom_names:
        datafr['Atom_names'] = datafr['Atom_names'].map(atoms_encoding)
        datafr[['atom_cos','atom_sin']] = pd.DataFrame(
        datafr['Atom_names'].tolist(), index=datafr.index)
        datafr = datafr.drop(columns=['Atom_names'])
    data_tensor = torch.tensor(datafr.values, dtype=torch.float32)

    data_tensor[:,0:3] = data_tensor[:,0:3]@rotate(rotation_angles[0], rotation_angles[1], rotation_angles[2]).T
    projected = project_to_isometric(data_tensor)
    
    
    max_of_any_coord = torch.max(data_tensor[:,0:2].flatten())
    min_of_any_coord = torch.min(data_tensor[:,0:2].flatten())

    zoom = zoom_val * res
    projected = zoom * projected + (res/2)


    grid = torch.ones((3,res, res), dtype=torch.float32)

    distance_to_camera = torch.stack([data_tensor[:,0]-max_point_mag, data_tensor[:,1]-max_point_mag, data_tensor[:,2]-max_point_mag],dim=1)
    magnitude = torch.sqrt(torch.sum(distance_to_camera**2, dim=1))

    #colours
    red_channel = (data_tensor[:,3]-min(data_tensor[:,3]))/(max(data_tensor[:,3])-min(data_tensor[:,3]))
    green_channel = (data_tensor[:,4]-min(data_tensor[:,4]))/(max(data_tensor[:,4])-min(data_tensor[:,4]))
    blue_channel = (magnitude-min(magnitude))/(max(magnitude)-min(magnitude))
    colours = torch.stack([red_channel, green_channel, blue_channel], dim=1)

    
    distances_3d = torch.cdist(data_tensor[:, 0:3], data_tensor[:, 0:3])
    distances_3d.fill_diagonal_(float('inf'))
    min_3d_values, _ = torch.min(distances_3d, dim=1)  
    

    sort_idx = torch.argsort(magnitude, descending=True)




    x = projected[:, 0].round().long() 
    y = projected[:, 1].round().long()
    r = (min_3d_values/2) * zoom
    x = x[sort_idx]
    y = y[sort_idx]
    r = r*1.28
    r = r[sort_idx].round().long()
    colours = colours[sort_idx]
    magnitude = magnitude[sort_idx]
    for i in range(len(x)):
        cx, cy, radius = x[i].item(), y[i].item(), r[i].item()
        color = colours[i] 
        if rad[0] == 1:
            radius = rad[1]
        else:
            radius = r[i].item()


        #Calculate a bounding box around the circle
        x_min = max(0, cx - radius)
        x_max = min(res - 1, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(res - 1, cy + radius)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Generate local pixel coordinates
        y_range = torch.arange(y_min, y_max + 1)
        x_range = torch.arange(x_min, x_max + 1)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')

        #Create a boolean circle mask based on the distance formula
        circle_mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
    
        #Paint the pixels across the color channels
        grid[:, y_grid[circle_mask], x_grid[circle_mask]] = color.unsqueeze(1)
    return grid










#Generate many images
def generate_images(data,number_of_images,speed,rad,atom_type,res=400,zoom=0.04,has_first_row_as_atom_names=False):
    video_tensor = torch.zeros(number_of_images,3,res,res)
    for i in range(number_of_images):
        random_rotation = torch.rand(3) * 2 * torch.pi
        video_tensor[i,:,:,:] =   process_data(data=data,
                                               naming_index=i,
                                               rotation_angles = random_rotation,
                                               rad=rad,
                                               atom_type=atom_type,
                                               res=res,
                                               zoom_val=zoom,
                                               has_first_row_as_atom_names=has_first_row_as_atom_names)
    return(video_tensor)

