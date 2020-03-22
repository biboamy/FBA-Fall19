import torch
from torch import nn, distributions
from torch.autograd import Variable


class PCPerformanceVAE(nn.Module):
    """
    Class to implement an encoder based on 1-d conv an gru units for performance pitch
    contours
    """

    def __init__(self, dropout_prob=0.5):
        """
        Initialized the network
        """
        super(PCPerformanceVAE, self).__init__()
        # initialize interal parameters
        self.conv_kernel_size = 7
        self.conv_stride = 3
        self.num_conv_features = 4
        self.num_recurrent_layers = 2
        self.z_dim = 32
        # define the different convolutional modules
        self.enc_conv_layers = nn.Sequential(
            # define the 1st convolutional layer
            nn.Conv1d(1, self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 2nd convolutional layer
            nn.Conv1d(self.num_conv_features, 2 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 3rd convolutional layer
            nn.Conv1d(2 * self.num_conv_features, 4 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 4th convolutional layer
            nn.Conv1d(4 * self.num_conv_features, 8 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )
        # define encoder recurrent layer
        self.enc_rnn = nn.GRU(
            8 * self.num_conv_features, 2 * self.z_dim, self.n_layers, batch_first=True
        )
        # define encoder mean and variance layers
        self.enc_mean = nn.Linear(2 * self.z_dim, self.z_dim)
        self.enc_log_std = nn.Linear(2 * self.z_dim, self.z_dim)

        # define decoder recurrent layer
        self.dec_rnn = nn.GRU(
            self.z_dim, 8 * self.num_conv_features, self.n_layers, batch_first=True
        )
        # define decoder conv layers
        self.dec_conv_layers = nn.Sequential(
            # define the 1st convolutional layer
            nn.ConvTranspose1d(8 * self.num_conv_features, 4 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 2nd convolutional layer
            nn.ConvTranspose1d(4 * self.num_conv_features, 2 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 3rd convolutional layer
            nn.ConvTranspose1d(2 * self.num_conv_features, self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 4th convolutional layer
            nn.ConvTranspose1d(self.num_conv_features, 1, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, input_tensor):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
            input_tensor:  torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                    mini_batch_size:    size of the mini batch during one training iteration
                    zero_pad_len:       length to which each input sequence is zero-padded
                    seq_lengths:        torch tensor (mini_batch_size x 1), length of each pitch contour
        """
        # get mini batch size from input and reshape input
        mini_batch_size, zero_pad_len = input_tensor.size()
        # initialize the hidden state
        self.init_hidden(mini_batch_size=mini_batch_size)
        # reshape input
        input_tensor = input_tensor.view(mini_batch_size, 1, zero_pad_len)

        # encode
        z_dist = self.encode(input_tensor)

        # reparametrize
        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        # compute output of decoding layer
        output = self.decode(z_tilde).view(input_tensor.size())

        return output, z_dist, prior_dist, z_tilde, z_prior

    def encode(self, x):
        # compute the output of the convolutional layer
        conv_out = self.enc_conv_layers(x)
        # compute the output of the lstm layer
        # transpose to ensure sequence length is dim 1 now
        lstm_out, self.hidden = self.enc_rnn(conv_out.transpose(1, 2))
        # compute the mean and variance
        z_mean = self.enc_mean(self.hidden)
        z_log_std = self.enc_log_std(self.hidden)
        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution

    def decode(self, z):
        pass

    @staticmethod
    def reparametrize(z_dist):
        """
        Implements the reparametrization trick for VAE
        """
        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
