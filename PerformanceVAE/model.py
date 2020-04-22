import torch
from torch import nn, distributions
from torch.autograd import Variable
from collections import OrderedDict


class PCPerformanceEncoder(nn.Module):
    """
    Class to implemnt an encoder based on pitch contours for performance assessment
    """
    def __init__(
            self,
            input_size=2000,
            num_in_channels=2,
            dropout_prob=0.5,
            z_dim=16,
            kernel_size=7,
            stride=1,
            num_conv_features=4
    ):
        super(PCPerformanceEncoder, self).__init__()
        # initialize interal parameters
        self.input_size = input_size
        self.conv_kernel_size = kernel_size
        self.num_in_channels = num_in_channels
        self.conv_stride = 1
        self.num_rnn_rollouts = 32
        self.num_conv_features = num_conv_features
        self.z_dim = z_dim

        # define the different convolutional modules
        self.kernel_size = 7
        self.stride = 3
        self.n0_features = 4
        self.n1_features = 8
        self.n2_features = 16
        self.enc_conv_layers = nn.Sequential(
            # define the 1st convolutional layer
            nn.Conv1d(self.num_in_channels, self.n0_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n0_features),
            nn.ReLU(),
            # nn.Dropout(),
            # define the 2nd convolutional layer
            nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n1_features),
            nn.ReLU(),
            # nn.Dropout(),
            # define the 3rd convolutional layer
            nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n2_features),
            nn.ReLU(),
            # nn.Dropout(),
            # define the final fully connected layer (fully convolutional)
            nn.Conv1d(self.n2_features, self.n2_features, self.kernel_size, self.stride),
            nn.BatchNorm1d( self.n2_features),
            nn.ReLU(),
            # nn.Dropout()
        )

        # define encoder linear layer
        self.out_seq_len = self.input_size - 4 * (self.conv_kernel_size - 1)
        self.enc_lin = nn.Linear(self.out_seq_len, self.num_rnn_rollouts)
        self.enc_lin2 = nn.Linear(8 * self.num_conv_features * self.num_rnn_rollouts, 2 * self.z_dim)

        self.enc_mean = nn.Linear(2 * self.z_dim, self.z_dim)
        self.hidden_size = 16
        self.n_layers = 1
        self.lstm = nn.GRU(self.n2_features, self.hidden_size, self.n_layers, batch_first=True)

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_tensor):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
            input_tensor:  torch Variable (mini_batch_size x in_channels x zero_pad_len), of input pitch contours
                    mini_batch_size:    size of the mini batch during one training iteration
                    in_channels:        number of input channels
                    zero_pad_len:       length to which each input sequence is zero-padded
        """
        if len(input_tensor.size()) == 2:
            input_tensor = input_tensor.unsqueeze(1)
        mini_batch_size, in_channels, zero_pad_len = input_tensor.size()
        c_out = self.enc_conv_layers(input_tensor)
        # final_output = torch.mean(torch.mean(c_out, 2), 1).view(-1, 1)
        lstm_out, self.hidden = self.lstm(c_out.transpose(1, 2))
        final_output = torch.mean(torch.mean(lstm_out, 1), 1).view(-1, 1)
        return final_output


class PCPerformanceVAE(nn.Module):
    """
    Class to implement an VAE based on 1-d conv and gru units for performance pitch
    contours
    """

    def __init__(
            self,
            input_size=2000,
            num_in_channels=1,
            dropout_prob=0.5,
            z_dim=16,
            kernel_size=7,
            stride=3,
            num_conv_features=4
    ):
        """
        Initialized the network
        """
        super(PCPerformanceVAE, self).__init__()
        # initialize interal parameters
        self.input_size = input_size
        self.conv_kernel_size = kernel_size
        self.conv_stride = 1
        self.num_rnn_rollouts = 32
        self.num_conv_features = num_conv_features
        self.z_dim = z_dim
        self.z_assess = self.z_dim // 2
        self.z_score = self.z_dim - self.z_assess

        # define the different convolutional modules
        self.enc_conv_layers = nn.Sequential(
            # define the 1st convolutional layer
            nn.Conv1d(1, self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.BatchNorm1d(self.num_conv_features),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),

            # define the 2nd convolutional layer
            nn.Conv1d(self.num_conv_features, 2 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.BatchNorm1d(2 * self.num_conv_features),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),

            # define the 3rd convolutional layer
            nn.Conv1d(2 * self.num_conv_features, 4 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.BatchNorm1d(4 * self.num_conv_features),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),

            # define the 4th convolutional layer
            nn.Conv1d(4 * self.num_conv_features, 8 * self.num_conv_features, self.conv_kernel_size, self.conv_stride),
            nn.BatchNorm1d(8 * self.num_conv_features),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
        )

        # define encoder linear layer
        self.out_seq_len = self.input_size - 4 * (self.conv_kernel_size - 1)
        self.enc_lin = nn.Linear(self.out_seq_len, self.num_rnn_rollouts)
        self.enc_lin2 = nn.Linear(8 * self.num_conv_features * self.num_rnn_rollouts, 2 * self.z_dim)

        # define encoder mean and variance layers
        self.enc_mean = nn.Linear(2 * self.z_dim, self.z_dim)
        self.enc_log_std = nn.Linear(2 * self.z_dim, self.z_dim)

        # define decoder linear layer
        self.dec_lin3 = nn.Linear(self.z_score, 2 * self.z_dim)
        self.dec_lin2 = nn.Linear(2 * self.z_dim, 8 * self.num_conv_features * self.num_rnn_rollouts)
        self.dec_lin = nn.Linear(self.num_rnn_rollouts, self.out_seq_len)

        # define decoder conv layers
        self.dec_conv_layers = nn.Sequential(
            # define the 1st convolutional layer
            nn.ConvTranspose1d(
                8 * self.num_conv_features, 4 * self.num_conv_features, self.conv_kernel_size, self.conv_stride
            ),
            nn.BatchNorm1d(4 * self.num_conv_features),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),

            # define the 2nd convolutional layer
            nn.ConvTranspose1d(4 * self.num_conv_features, 2 * self.num_conv_features, self.conv_kernel_size,
                               self.conv_stride),
            nn.BatchNorm1d(2 * self.num_conv_features),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),

            # define the 3rd convolutional layer
            nn.ConvTranspose1d(2 * self.num_conv_features, self.num_conv_features, self.conv_kernel_size,
                               self.conv_stride),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),

            # define the 4th convolutional layer
            nn.ConvTranspose1d(self.num_conv_features, 1, self.conv_kernel_size, self.conv_stride),
            nn.BatchNorm1d(1),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.z_assess, 2 * self.z_assess),
            nn.SELU(),
            nn.Linear(2 * self.z_assess, 1),
            nn.SELU()
        )

        # self.xavier_initialization()

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_tensor):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
            input_tensor:  torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                    mini_batch_size:    size of the mini batch during one training iteration
                    zero_pad_len:       length to which each input sequence is zero-padded
        """
        # get mini batch size from input and reshape input
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
        mini_batch_size, zero_pad_len = input_tensor.size()

        # reshape input
        input_tensor = input_tensor.view(mini_batch_size, 1, zero_pad_len)

        # encode
        z_tilde = self.encode(input_tensor)
        # z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        # compute performance score from latent code
        performance_score = self.classifier(z_tilde[:, :self.z_assess])  # batch x 1

        # DECODE
        # pass through decoder
        output = self.decode(z_tilde[:, self.z_assess:])
        output = output.view(input_tensor.size())

        return output, performance_score   # ,z_dist, prior_dist

    def encode(self, x):
        # NAN Check
        for name, param in self.named_parameters():
            if 'weight' in name:
                nan_check = torch.isnan(param.data)
                if nan_check.nonzero().size(0) > 0:
                    print(f'Layer {name} has become nan')
                    raise ValueError

        # compute the output of the convolutional layer
        mini_batch_size = x.shape[0]
        c_out = self.enc_conv_layers(x)
        c_out = nn.functional.selu(self.enc_lin(c_out))
        c_out = nn.functional.selu(self.enc_lin2(c_out.view(mini_batch_size, -1)))
        # compute the mean and variance
        z_mean = self.enc_mean(c_out)
        # z_log_std = self.enc_log_std(c_out)
        # (num_layers * num_directions, batch, hidden_size)
        # z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_mean

    def decode(self, dec_in):
        dec_out = nn.functional.selu(self.dec_lin3(dec_in))
        dec_out = nn.functional.selu(self.dec_lin2(dec_out))
        dec_out = dec_out.contiguous().view(dec_out.shape[0], -1, self.num_rnn_rollouts)
        dec_out = nn.functional.selu(self.dec_lin(dec_out))
        dec_out = self.dec_conv_layers(dec_out)
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
