import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader


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


def _get_pretrained_model(model_name, hidden_units, dropout):
    '''Gets a pretrained model from pytorch, freezes the majority of layers in 
    it, and then adds a sequential classifer

    Inputs:
        model_name: A string descriptor of which PyTorch classifier is being 
        imported. Only 'vgg16', 'resnet50', and 'googlenet' are supported.

        hidden_units: intermediary number of units in the sequential 
        classifier we are adding.

        dropout: dropout percentage in the sequential classifier we are 
        adding

    Outputs:
        A PyTorch model that only has a small fraction of its layers open to 
        training
    '''
    if model_name == "vgg16":
        model = models.vgg16(pretrained = True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained = True)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained = True)
    else:
        raise NotImplementedError

    for param in model.parameters():
        param.requires_grad = False

    if model_name == "vgg16":
        num_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = hidden_units), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(in_features = hidden_units, out_features = 120), 
            nn.LogSoftmax(dim = 1)
            )
    else:
        num_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = hidden_units), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(in_features = hidden_units, out_features = 120), 
            nn.LogSoftmax(dim = 1)
            )
    return model


def _get_train_data_loader(batch_size, training_dir):
    '''Sets up an ImageFolder within the training directory and specifies the 
    transformations that will happen 
    '''
    print("Getting train data loader.")
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size = 256, scale = (0.85, 1.0)),
            transforms.RandomRotation(degrees = 20),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    train_images = datasets.ImageFolder(root = training_dir,
            transform = train_transform)
    return DataLoader(dataset = train_images, batch_size=batch_size,
         shuffle = True)


def train(model, train_loader, epochs, criterion, optimizer, device):
    '''
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    '''
    
    # Create a piece of the model that translates from labels to names
    model.class_to_idx = train_loader.dataset.class_to_idx
    model.idx_to_class = {
        idx: class_ for class_, idx in model.class_to_idx.items()
    }

    # training loop
    for epoch in range(1, epochs + 1):
        
        train_loss = 0.0
        train_acc = 0.0
        model.train() # Make sure that the model is in training mode.

        for ii, (data, target) in enumerate(train_loader):
            # get data
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # get log probabilities from model
            output = model(data)
            
            # perform backprop
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()
            
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
        
        epoch_str = "Epoch {}".format(epoch)
        loss_str = "Loss {:.4f}".format(train_loss / len(train_loader.dataset))
        acc_str = "Accuracy: {:.2%}".format(train_acc / len(train_loader.dataset))
        print(", ".join([epoch_str, loss_str, acc_str]))


if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Adds arguments for the function needed to load pretrained model
    parser.add_argument('--model_name', type=str,  default='vgg16', metavar='N',
                        help='which type of pretrained model is being loaded')
    parser.add_argument('--hidden_units', type=int, default=512, metavar='N', 
                        help='dimension of hidden layer (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.33, metavar='N', 
                        help='dropout within hidden layer (default: 0.33)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = _get_pretrained_model(model_name = args.model_name,
         hidden_units = args.hidden_units, 
         dropout = args.dropout)
    model.to(device)

    # Use ADAM as your optimizer and Negative Log Likelihood as your loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    
    # Saves the kewyword arguments for loading a pretrained model in model_info
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "class_to_idx": model.class_to_idx,
            "idx_to_class": model.idx_to_class,
            "model_name": args.model_name,
            "hidden_units": args.hidden_units,
            "dropout": args.dropout
        }
        torch.save(model_info, f)
    
	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)