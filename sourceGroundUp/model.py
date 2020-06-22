import torch.nn as nn
import torch.nn.functional as F 

class BasicConvNet(nn.Module):
    
    '''This is a simple convolutional neural network with two groups of 
    convolution -> ReLU -> maxpooling layers that then pass things on to two 
    fully connected layers that handle the classification steps. Meant to serve 
    as a more intelligent benchmark than our probabilistic random guesser.

    General sequence of layers I used was inspired by the following paper:
    https://jgeekstudies.org/2017/03/12/who-is-that-neural-network/

    Guidelines on how to select hyperparameters for the convolution and pooling 
    layers came from:
    https://cs231n.github.io/convolutional-networks/
    '''

    def __init__(self):
        '''Not including many tunable parameters in this class because I 
        want to be sure the architecture works. 

        The main thing I needed to make sure of here: since we're using
        Negative Log Likelihood as our loss function, we need to apply
        a LogSoftmax

        '''
        super(BasicConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16,
             kernel_size = 3, stride = 1,  padding = 1, 
             padding_mode = "zeros")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2,
             padding = 0)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16,
             kernel_size = 3, stride = 1,  padding = 1,
             padding_mode = "zeros")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2,
             padding = 0)

        self.fc1 = nn.Linear(in_features = 56 * 56 * 16, 
             out_features = 56 *56)
        self.fc2 = nn.Linear(in_features = 56 * 56, 
             out_features = 120)
        self.logsoft = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        '''Relatively quick sizing overview

        conv1: Input - (224, 224, 3), Output - (224, 224, 16)
        ReLU
        pool1: Input - (224, 224, 16), Output - (112, 112, 16)

        conv2: Input - (112, 112, 16), Output - (112, 112, 16)
        ReLU
        pool2: Input - (112, 112, 16), Output - (56, 56, 16)

        Then the input is flattened down before being passed to the fully 
        connected layers

        fc1: Input - (50176,), Output - (3136,)
        ReLU
        fc2: Input - (3136,), Output - (120,)
        '''
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 56 * 56 * 16)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return self.logsoft(x)
