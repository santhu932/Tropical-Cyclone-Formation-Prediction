import torch
import torch.nn as nn
from rss_lstm.sam_convLSTMCell import sam_ConvLSTMCell


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SAMConvLSTM(nn.Module):

    def __init__(self, attention_hidden_dims, in_channels, out_channels,
    kernel_size, padding, activation, frame_size, bias, w_init):

        super(SAMConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.bias = bias
        self.w_init = w_init

        # We will unroll this over time steps
        self.sam_convLSTMcell = sam_ConvLSTMCell(attention_hidden_dims, in_channels, out_channels,
        kernel_size, padding, activation, frame_size, bias = self.bias, w_init=self.w_init)
        

    def forward(self, X, previous_H = None, previous_C = None, previous_M = None):

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
            
        if previous_M == None:
            M = torch.zeros(batch_size,self.out_channels, height, width, device=device)
        else:
            M = previous_M.to(device)

        # Unroll over time steps
        
        H, C, M = self.convLSTMcell(X, H, C, M)

        output = H

        return output, C, M

        
