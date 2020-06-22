import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# local import to get our model class. CHANGE .models
from models import BasicConvNet


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
    model = BasicConvNet()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model


def _get_train_data_loader(batch_size, training_dir):
    '''Sets up an ImageFolder within the training directory and specifies the 
    transformations that will happen 
    '''
    print("Getting train data loader.")
    image_transforms = {
        "train":
        transforms.Compose([
            transforms.RandomResizedCrop(size = 256, scale = (0.85, 1.0)),
            transforms.RandomRotation(degrees = 20),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test":
        transforms.Compose([
            transforms.Resize(size = 256),
            transforms.CenterCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    testing_dir = str(training_dir).replace("train","test")
    data = {
        "train": datasets.ImageFolder(root = training_dir,
            transform = image_transforms["train"]),
        "test": datasets.ImageFolder(root = testing_dir, 
            transform = image_transforms["test"])
    }
    return DataLoader(dataset = data["train"], batch_size=batch_size,
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
    
    # training loop is provided
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

        print(f"Epoch: {epoch}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_acc / len(train_loader):.2%}")


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
        
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = BasicConvNet()
    model.to(device)

    # Use ADAM as your optimizer and Negative Log Likelihood as your loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    ## TODO: complete in the model_info by adding three argument names, the first is given
    # Keep the keys of this dictionary as they are 
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        torch.save(model_info, f)
    
	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)