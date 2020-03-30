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
        self.n_layers = 1

        # define the different convolutional modules
        self.enc_conv_layers = nn.Sequential(
            # define the 1st convolutional layer (batch, ncf, 665)
            nn.Conv1d(1, self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 2nd convolutional layer (batch, ncf, 220)
            nn.Conv1d(self.num_conv_features, 2 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 3rd convolutional layer (batch, ncf, 72)
            nn.Conv1d(2 * self.num_conv_features, 4 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 4th convolutional layer (batch, ncf, 22)
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

        # define linear layer for input to decoder rnn
        self.dec_in_linear = nn.Linear(self.z_dim, 8 * self.num_conv_features)
        # define decoder recurrent layer
        self.dec_rnn = nn.GRU(
            self.z_dim, 8 * self.num_conv_features, self.n_layers, batch_first=True
        )
        # define decoder conv layers
        self.dec_conv_layers = nn.Sequential(
            # define the 1st convolutional layer
            nn.ConvTranspose1d(8 * self.num_conv_features, 4 * self.num_conv_features, self.conv_kernel_size, self.conv_stride, output_padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 2nd convolutional layer
            nn.ConvTranspose1d(4 * self.num_conv_features, 2 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 3rd convolutional layer
            nn.ConvTranspose1d(2 * self.num_conv_features, self.num_conv_features, self.conv_kernel_size, self.conv_stride, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # define the 4th convolutional layer
            nn.ConvTranspose1d(self.num_conv_features, 1, self.conv_kernel_size, self.conv_stride, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim*5),
            nn.ReLU(),
            nn.Linear(self.z_dim*5, 1)
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
        input_size = input_tensor.size()
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
        mini_batch_size, zero_pad_len = input_tensor.size()
        # initialize the hidden state
        self.init_hidden(mini_batch_size=mini_batch_size)
        # reshape input
        input_tensor = input_tensor.view(mini_batch_size, 1, zero_pad_len)

        # encode
        z_dist, seq_len = self.encode(input_tensor)
        # reparametrize
        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        # compute performance score from latent code
        performance_score = self.classifier(z_tilde)  # batch x 1

        # compute output of decoding layer
        # create input for decoder rnn
        dec_lstm_in = self.dec_in_linear(z_tilde).unsqueeze(1).repeat([1, seq_len, 1])  # batch x seq_len x z_dim
        output = self.decode(z_tilde.unsqueeze(0), dec_lstm_in).view(input_size)

        return output, performance_score, z_dist, prior_dist, z_tilde, z_prior

    def encode(self, x):
        # compute the output of the convolutional layer
        conv_out = self.enc_conv_layers(x).transpose(1, 2)
        seq_len = conv_out.shape[1]
        # compute the output of the lstm layer
        # transpose to ensure sequence length is dim 1 now
        _, hidden = self.enc_rnn(conv_out, self.hidden)
        hidden = torch.squeeze(hidden, 0)
        # compute the mean and variance
        z_mean = self.enc_mean(hidden)
        z_log_std = self.enc_log_std(hidden)
        #  (num_layers * num_directions, batch, hidden_size)

        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution, seq_len

    def decode(self, z, inp):
        dec_out, _ = self.dec_rnn(inp, z)
        dec_out = self.dec_conv_layers(dec_out.transpose(1, 2))
        return dec_out

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
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, 2 * self.z_dim))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
