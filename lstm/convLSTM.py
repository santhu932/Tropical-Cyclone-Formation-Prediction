import torch
import torch.nn as nn
from lstm.convLSTMCell import ConvLSTMCell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels,
    kernel_size, padding, activation, frame_size, bias, w_init):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.bias = bias
        self.w_init = w_init

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
        kernel_size, padding, activation, frame_size, bias = self.bias)
        
        if self.w_init == True:
            self.convLSTMcell.apply(self.weights_init)
        
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LeakyReLU):
            m.negative_slope = 0.1

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
        height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels,
        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output

        
