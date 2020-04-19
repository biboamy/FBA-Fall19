import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        momentum = 0.5
        self.bn1 = nn.BatchNorm2d(inp, momentum=momentum)       
        self.conv1 = nn.Conv2d(inp, out, (3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out, momentum=momentum)       
        self.conv2 = nn.Conv2d(out, out, (3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(out, momentum=momentum)       

        self.sk = nn.Conv2d(inp, out, (1,1), padding=(0,0))
    def forward(self, x):
        out = self.conv1(self.bn1(x))
        #out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))
        #out = self.conv2(F.relu(out))
        out = self.bn3(out)
        #out += self.sk(x)
        return out

class ConvNet_Fixed(nn.Module): # The same ConvNet with fixed input size

    def __init__(self, model_name):
        """
        Initializes the class with internal parameters for the different layers
        Args:
            mode:       int, 0,1 specifying different minimum input size, 0: 1000, 1:500
        """
        super(ConvNet_Fixed, self).__init__()
        # initialize model internal parameters
        self.kernel_size = 3
        self.stride = 3
        self.n0_features = 4
        self.n1_features = 4
        self.n2_features = 4
        # define the different convolutional modules
        
        if 'Residual' in model_name:
            # Residual CNN
            self.conv = nn.Sequential(
                    nn.Conv2d(1, 4, (3,3), (3,3)),
                    block(4, 4),
                    nn.Dropout(p=0.2),
                    nn.MaxPool2d((3,3),(3,3)),
                    block(4, 4),
                    nn.Dropout(p=0.2),
                    nn.MaxPool2d((3,3),(3,3)),
                    block(4, 4),
                    nn.Dropout(p=0.2),
                    #nn.InstanceNorm2d(4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d( 4, 4, (3,3), (3,3))
                )
        else:
            self.conv = nn.Sequential(
                # define the 1st convolutional layer
                nn.Conv2d(1, self.n0_features, self.kernel_size, self.stride),  # output is (1000 - 7)/3 + 1 = 332
                nn.ReLU(),
                # nn.Dropout(),
                # define the 2nd convolutional layer
                nn.Conv2d(self.n0_features, self.n1_features, self.kernel_size, self.stride),
                # output is (332 - 7)/3 + 1 = 109
                nn.ReLU(),
                # nn.Dropout(),
                # define the 3rd convolutional layer
                nn.Conv2d(self.n1_features, self.n2_features, self.kernel_size, self.stride),
                # output is (109 - 7)/3 + 1 = 35
                nn.ReLU(),
                # nn.Dropout(),
                # define the final fully connected layer (fully convolutional)
                nn.Conv2d(self.n2_features, self.n2_features, self.kernel_size, self.stride),
                nn.ReLU(),
                #nn.Dropout()
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(196, 128), # first dim need to be determined
            nn.ReLU(),
            nn.Dropout(),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            nn.Linear(128, 1)
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
        
        mini_batch_size, dim1, dim2, dim3 = input.shape
        # print(input.shape)
        oup = self.conv(input)
        #oup = oup.flatten()
        oup = oup.view(mini_batch_size, -1)
        oup = self.classifier(oup)
        oup = torch.sigmoid(oup)
       
        return oup

class Net_Fixed(nn.Module):
    def __init__(self, model_name):
        super(Net_Fixed, self).__init__()
        self.model = ConvNet_Fixed(model_name)

    def forward(self, matrix):

        oup = self.model(matrix)
        return oup