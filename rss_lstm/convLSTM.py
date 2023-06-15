import torch
import torch.nn as nn
from rss_lstm.convLSTMCell import ConvLSTMCell

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

    def forward(self, X, previous_H = None, previous_C = None):

        # X is a frame sequence (batch_size, num_channels, height, width)

        # Get the dimensions
        batch_size, _, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels,
        height, width, device=device)
        
        # Initialize Hidden State
        if previous_H == None:
            H = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        else:
            H = previous_H.to(device)

        # Initialize Cell Input
        if previous_C == None:
            C = torch.zeros(batch_size,self.out_channels, height, width, device=device)
        else:
            C = previous_C.to(device)

        # Unroll over time steps
        
        H, C = self.convLSTMcell(X, H, C)

        output = H

        return output, C

        
