import torch
import torch.nn as nn
from rss_lstm.self_attention import SelfAttention
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from rss_lstm.convLSTMCell import ConvLSTMCell

class sam_ConvLSTMCell(nn.Module):

    def __init__(self, attention_hidden_dims, in_channels, out_channels,
    kernel_size, padding, activation, frame_size, bias, w_init):

        super(sam_ConvLSTMCell, self).__init__()
        self.bias = bias
        self.w_init = w_init
            
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
        kernel_size, padding, activation, frame_size, bias = self.bias)
        self.attention_x = SelfAttention(in_channels, attention_hidden_dims)
        self.attention_h = SelfAttention(out_channels, attention_hidden_dims)
        
        if self.w_init == True:
            self.convLSTMcell.apply(self.weights_init)
            self.attention_x.apply(self.weights_init)
            self.attention_h.apply(self.weights_init)
            
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
        
        
    def forward(self, X, H_prev, C_prev):
        X, _ = self.attention_x(X)
        new_h, new_cell = self.convLSTMcell(X, H_prev, C_prev)
        new_h, attention_h = self.attention_h(new_h)
        new_h += new_h
        return new_h, new_cell
