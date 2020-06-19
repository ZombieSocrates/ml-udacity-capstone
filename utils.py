import numpy as np 
import pathlib


def get_breed_name(nested_file_arr):
    '''Extracts the breed name associated with a particular image from the 
    nested numpy arrays in the parsed .mat files

    Input:
        nested_file_arr: numpy array of numpy arrays, each containing
        a single item like n02085620-Chihuahua/n02085620_10131.jpg'

    Output:
        the string representing the class name associated with that file 
        (e.g. Chihuahua)
    '''
    return [x.item(0).split("/")[0].split("-")[1] for x in nested_file_arr]


def map_breed_labels_to_names(file_list):
    '''Returns a mapping from the numeric labels in the Stanford Dogs data
    set to each of the 120 breed names.

    Input:
        file_list: any of the `file_list` objects within the downloaded
        and parsed .mat files.

    Output:
        a dictionary with 120 breed_labe; : breed_name entries
    '''
    out_dict = {}
    name_list = np.apply_along_axis(get_breed_name, 0, file_list)
    unique_names, orig_idx = np.unique(name_list, return_index = True)
    for i, name in enumerate(unique_names[np.argsort(orig_idx)]):
        out_dict[i + 1] = name
    return out_dict


def get_image_path(file_name_arr, root_dir):
        '''Given one of the numpy arrays within the 'file_list' object of 
    test_list.mat or train_list.mat, extracts the matching annotation from 
    `annotation_list'

    Input:
        file_name_arr: numpy array that contains a single image path.

        root_dir: the name of the directory where you downloaded and extracted 
        the data set. 

    Output:
        A Pathlib path that points to the unprocessed image
    '''
    file_path = file_name_arr.item()
    return pathlib.Path(root_dir) / "Images" / file_path


def get_annotation_path(file_name_arr, root_dir = data_dir):
    '''Given one of the numpy arrays within the 'file_list' object of 
    test_list.mat or train_list.mat, extracts the matching annotation from 
    `annotation_list'

    Input:
        file_name_arr: numpy array that contains a single image path.

        root_dir: the name of the directory where you downloaded and extracted 
        the data set. 

    Output:
        A Pathlib path that points to the annotation file
    '''
    file_path = pathlib.Path(file_name_arr.item())
    annot_file_name = file_path.stem
    breed_dir = file_path.parent.stem
    return root_dir / "Annotation" / breed_dir / annot_file_name