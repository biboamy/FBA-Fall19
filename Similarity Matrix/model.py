import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvNet(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
     pitch contours as input
    """

    def __init__(self):
        """
        Initializes the class with internal parameters for the different layers
        Args:
            mode:       int, 0,1 specifying different minimum input size, 0: 1000, 1:500
        """
        super(ConvNet, self).__init__()
        # initialize model internal parameters
        self.kernel_size = 3
        self.stride = 3
        self.n0_features = 4
        self.n1_features = 8
        self.n2_features = 16
        # define the different convolutional modules
        self.conv = nn.Sequential(
            # define the 1st convolutional layer
            nn.Conv2d(1, self.n0_features, self.kernel_size, self.stride),# output is (1000 - 7)/3 + 1 = 332
            nn.ReLU(),
            #nn.Dropout(),
            # define the 2nd convolutional layer
            nn.Conv2d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (332 - 7)/3 + 1 = 109
            nn.ReLU(),
            #nn.Dropout(),
            # define the 3rd convolutional layer
            nn.Conv2d(self.n1_features, self.n2_features, self.kernel_size, self.stride), # output is (109 - 7)/3 + 1 = 35
            nn.ReLU(),
            #nn.Dropout(),
            # define the final fully connected layer (fully convolutional)
            nn.Conv2d(self.n2_features, self.n2_features, self.kernel_size, self.stride),
            nn.ReLU(),
            #nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1)
        )

    def forward(self, input):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
                input:  torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                        mini_batch_size:    size of the mini batch during one training iteration
                        zero_pad_len:       length to which each input sequence is zero-padded
                        seq_lengths:        torch tensor (mini_batch_size x 1), length of each pitch contour
        """
        # get mini batch size from input and reshape
        oup = self.conv(input)
        oup = torch.mean(torch.mean(oup, 2),2)
        oup = self.classifier(oup)
        
        return oup

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = ConvNet()

    def forward(self, matrix):

        oup = self.model(matrix)
        return oup
