import numpy as np 
import pathlib
import xmltodict

from PIL import Image
from tqdm import tqdm


def get_breed_name(file_list_arr):
    '''Extracts the breed name associated with a particular image from the 
    nested numpy arrays in the parsed .mat files

    Input:
        file_list_arr: numpy array of numpy arrays, each containing
        a single item like n02085620-Chihuahua/n02085620_10131.jpg'

    Output:
        the string representing the class name associated with that file 
        (e.g. Chihuahua)
    '''
    return [x.item(0).split("/")[0].split("-")[1] for x in file_list_arr]


def map_breed_labels_to_names(file_list_arr):
    '''Returns a mapping from the numeric labels in the Stanford Dogs data
    set to each of the 120 breed names.

    Input:
        file_list_arr: any of the `file_list` objects within the downloaded
        and parsed .mat files.

    Output:
        a dictionary with 120 breed_labe; : breed_name entries

    TODO: Uncertain if this will actually be used outside of EDA. Maybe
    remove?
    '''
    out_dict = {}
    name_list = np.apply_along_axis(get_breed_name, 0, file_list_arr)
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
        the data set. Can be either a string or a pathlib PosixPath. 

    Output:
        A Pathlib path that points to the unprocessed image
    '''
    if not isinstance(root_dir, pathlib.PosixPath):
        root_dir = pathlib.Path(root_dir)
    file_path = file_name_arr.item()
    return root_dir / "Images" / file_path


def get_annotation_path(file_name_arr, root_dir):
    '''Given one of the numpy arrays within the 'file_list' object of 
    test_list.mat or train_list.mat, extracts the matching annotation from 
    `annotation_list'

    Input:
        file_name_arr: numpy array that contains a single image path.

        root_dir: the name of the directory where you downloaded and extracted 
        the data set. Can be either a string or a pathlib PosixPath. 

    Output:
        A Pathlib path that points to the annotation file
    '''
    if not isinstance(root_dir, pathlib.PosixPath):
        root_dir = pathlib.Path(root_dir)
    file_path = pathlib.Path(file_name_arr.item())
    annot_file_name = file_path.stem
    breed_dir = file_path.parent.stem
    return root_dir / "Annotation" / breed_dir / annot_file_name


def coord_tuple(bbox_dict):
    '''Converts bounding box dictionaries within the Annotation xml files into 
    tuples that specify the upper-left and bottom right corners surroundig a 
    dog

    Input:
        bbox_dict: a dictionary

    Output:
        A four-tuple that can be passed directly to the crop utility in Pillow
    '''
    left = int(bbox_dict["xmin"])
    upper = int(bbox_dict["ymin"])
    right = int(bbox_dict["xmax"])
    lower = int(bbox_dict["ymax"])
    return (left, upper, right, lower)


def extract_bbox_coords(annot_path):
    '''Given the path to the annotation xml files, return the coordinates 
    needed to crop out the non-dog parts of the image.

    Input:
        annot_path: a PosixPath returned by get_annotation_path() above

    Output:
        If the image has one and only one dog in it, returns aa tuple
        for the box bounding that dog. Otherwise returns a list of
        tuples bounding each dog present in the image.
    '''
    with open(annot_path, "r") as fd:
        raw_xml = xmltodict.parse(fd.read())
    coord_info = raw_xml["annotation"]["object"]
    if isinstance(coord_info, list):
        return [coord_tuple(v["bndbox"]) for v in coord_info]
    return coord_tuple(coord_info["bndbox"])


def get_class_dir(file_name_arr, root_dir, dataset_type):
    '''Given one of the numpy arrays within the 'file_list' object of 
    test_list.mat or train_list.mat, creates a new directory to hold the image 
    one it's been cropped down. Allows the data to be moved to S3 in a way 
    that PyTorch can play nicely with.

    Input:
        file_name_arr: numpy array that contains a single image path.

        root_dir: the name of the directory where you downloaded and extracted 
        the data set. Can be either a string or a pathlib PosixPath.

        dataset_type: a string representing whether the image specified by
        `file_name_arr` is in the train, test, or validation set

    Output:
        a string that can be easily turned into a directory
    '''
    if dataset_type not in ["train","valid","test"]:
        raise NotImplementedError("Put image in `train`, `test`, or `valid` folder...")
    if not isinstance(root_dir, pathlib.PosixPath):
        root_dir = pathlib.Path(root_dir)
    class_name = get_breed_name(file_name_arr)[0]
    return root_dir / dataset_type / class_name


def crop_and_ensure_RGB(img_obj, bbox_tuple):
    '''Given an image and a four-tuple bounding box, returns the bounded 
    subsection of that image, compressing to RGB if necessary

    Input:
        img_obj: a PIL image

        root_dir: a (left, upper, right, lower) tuple like the ones returned 
        by extract_bbox_coords()

    Output:
        the cropped PIL image
    '''
    cropped = img_obj.crop(box = bbox_tuple)
    if cropped.mode != "RGB":
        print("Applying RGB compression")
        cropped = cropped.convert("RGB")
    return cropped


def prepare_image_folder(file_list_arr, root_dir, 
    dataset_type, log_multiples = True):
    '''Parses to file information in the raw .mat files downloaded and sets up 
    the .gitignored `data` folder in a manner recognizable by PyTorch's 
    ImageFolder class. Assumed that you have extracted the raw dataset 
    files directly into the `data` folder

    Input:
        file_list_arr: any of the `file_list` objects within the downloaded
        train or test list .mat files.

        root_dir: the name of the directory where you downloaded and extracted 
        the data set. Can be either a string or a pathlib PosixPath.

        dataset_type: a string representing whether the image specified by
        `file_name_arr` is in the train, test, or validation set

        log_multiples: a debug / info-logging parameter that alerts you
        when multiple bounding boxes are found in a single image file

    Output:
        Doesn't return anything, just organizes your root_dir as 
        follows, using cropped images instead of entire ones.

        data/
            train/
                 breed-01/
                     file.jpg
                     file.jpg
                 breed-02

            test/
                 breed-01
                 breed-02
    '''
    mult_count = 0
    mult_img = 0
    for file_name_arr in tqdm(file_list_arr.flatten()):
        class_dir = get_class_dir(file_name_arr, root_dir, dataset_type)
        img_path = get_image_path(file_name_arr, root_dir)
        annot_path = get_annotation_path(file_name_arr, root_dir)
        bound_box = extract_bbox_coords(annot_path)
        if not class_dir.exists():
            class_dir.mkdir(parents = True, exist_ok = True)
        raw_img = Image.open(img_path)
        if isinstance(bound_box, tuple):
            sv_img = crop_and_ensure_RGB(raw_img, bound_box)
            sv_img.save(class_dir / img_path.name)
            continue
        if log_multiples:
            N = len(bound_box)
            mult_count += N
            mult_img += 1
            breed = class_dir.name
            print(f"Found {N} {breed}s in {img_path.name}")
        for i, bbox in enumerate(bound_box):
            sv_img = crop_and_ensure_RGB(raw_img, bbox)
            sv_name = f"{img_path.stem}-{i+1}{img_path.suffix}"
            sv_img.save(class_dir / sv_name)
    if log_multiples:
        print(f"\tTotal of {mult_count} multiples found in {mult_img} images")
    print(f"Finished preparing {dataset_type} data!!")
