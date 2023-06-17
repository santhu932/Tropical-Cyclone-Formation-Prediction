import torch
import torch.nn as nn
from rss_lstm.self_attention_memory_module import SelfAttentionMemory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from rss_lstm.convLSTMCell import ConvLSTMCell

class sam_ConvLSTMCell(nn.Module):

    def __init__(self, attention_hidden_dims, in_channels, out_channels,
    kernel_size, padding, activation, frame_size, bias, w_init):

        super(sam_ConvLSTMCell, self).__init__()
        self.bias = bias
            
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
        kernel_size, padding, activation, frame_size, bias = self.bias)
        self.attention_memory = SelfAttentionMemory(out_channels, attention_hidden_dims)
        
        if self.w_init == True:
            self.convLSTMcell.apply(self.weights_init)
            self.attention_memory.apply(self.weights_init)
            
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
        
        
    def forward(self, X, H_prev, C_prev, M_prev):

        new_h, new_cell = self.convLSTMcell(X, H_prev, C_prev)
        new_h, new_memory, attention_h = self.attention_memory(new_h, M_prev)
        

        return new_h.to(device), new_cell.to(device), new_memory.to(device)
