# import libraries
import os
import numpy as np
import torch
from six import BytesIO

from train import _get_pretrained_model

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'


# Provided model load function
def model_fn(model_dir):
    '''Load the PyTorch model from the `model_dir` directory.'''
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_pretrained_model(model_name = model_info["model_name"],
         hidden_units = model_info["hidden_units"],
         dropout = model_info["dropout"])
    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model


# Provided input data loading
def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Modified predict function
def predict_fn(input_data, model, topk = 5):
    '''Moves serialized numpy input to available GPU, converts it back to a
    Torch tensor, applies our model to the data, and gets back class labels
    and associated probabilities for that input. Will return the N most likely 
    classes.

    Inputs:

        input_data: a numpy array that has already had ImageFolder transforms
        applied to it (AKA, passed in from a DataLoader)

        model: the PyTorch model from model_fn

        topk: controls how many of the most likely classes will be returned by
        the model

    Output:
        a numpy array with two sub arrays, the first being the class labels 
        and the second being the associated probabilities
    '''
    print('Predicting top {} class labels for the input data...'.format(topk))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)
    model.eval()
    out = model(data)
    ps = torch.exp(out)
    top_ps, top_class = ps.topk(topk, dim = 1)
    classes_np = top_class.cpu().detach().numpy()
    probs_np = top_ps.cpu().detach().numpy()
    out_array = np.array([classes_np, probs_np])
    return out_array