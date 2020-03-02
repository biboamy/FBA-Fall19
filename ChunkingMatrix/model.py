import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *

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
            nn.Linear(100, 128), # first dim need to be determined
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


class ConvNet_LSTM(nn.Module):  # The same ConvNet with fixed input size

    def __init__(self, model_name):
        """
        Initializes the class with internal parameters for the different layers
        Args:
            mode:       int, 0,1 specifying different minimum input size, 0: 1000, 1:500
        """
        super(ConvNet_LSTM, self).__init__()
        # initialize model internal parameters
        self.kernel_size = 3
        self.stride = 3
        self.n0_features = 4
        self.n1_features = 4
        self.n2_features = 4
        self.hidden_size = 16
        self.n_layers = 1
        self.lstm_dim = 9 # change according to the input dim

        # define the different convolutional modules

        if 'Residual' in model_name:
            # Residual CNN
            self.conv = nn.Sequential(
                nn.Conv2d(1, 4, (3, 3), (3, 3)),
                block(4, 4),
                nn.Dropout(p=0.2),
                nn.MaxPool2d((3, 3), (3, 3)),
                block(4, 4),
                nn.Dropout(p=0.2),
                nn.MaxPool2d((3, 3), (3, 3)),
                block(4, 4),
                nn.Dropout(p=0.2),
                # nn.InstanceNorm2d(4),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, (3, 3), (3, 3))
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
                # nn.Dropout()
            )

        # self.classifier = nn.Sequential(
        #     nn.Linear(100, 128),  # first dim need to be determined
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     # nn.Linear(1024, 1024),
        #     # nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

        # define the LSTM module
        self.lstm = nn.GRU(self.n2_features * self.lstm_dim, self.hidden_size,
                           self.n_layers, batch_first=True)
        # intialize the hidden state
        self.init_hidden(1)
        # define the final linear layer
        self.linear = nn.Linear(self.hidden_size, 1)

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

        mini_batch_size, _, dim1, dim2, dim3 = input.shape
        #print(input.shape) # (32, 1, 9, 180, 180)

        oup = torch.zeros(mini_batch_size, dim1, self.n2_features*self.lstm_dim).cuda()
        for i in range(dim1):
            input_sub = input[:, :, i, :, :]
            oup_sub = self.conv(input_sub)
            oup_sub = oup_sub.view(mini_batch_size, -1)
            # print(oup_sub.shape)
            oup[:, i, :] = oup_sub

        # compute the output of the lstm layer
        #print(oup.shape)
        lstm_out, self.hidden = self.lstm(oup)
        #print(lstm_out.shape, self.hidden.shape)
        # extract final output of the lstm layer
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()
        final_lstm_out = lstm_out[:, lstm_seq_len - 1, :]
        # compute output of the linear layer
        #print(final_lstm_out.shape)
        oup = F.relu(self.linear(final_lstm_out))

        #oup = torch.sigmoid(oup)

        return oup

    def init_hidden(self, mini_batch_size):
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()

class Net_LSTM(nn.Module):
    def __init__(self, model_name):
        super(Net_LSTM, self).__init__()
        self.model = ConvNet_LSTM(model_name)

    def forward(self, matrix):

        oup = self.model(matrix)
        return oup