import torch.nn as nn
import torch
from lstm.convLSTM import ConvLSTM
from vit_pytorch import ViT
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
    activation, frame_size, num_layers, bias, w_init, hidden_size):

        super(Seq2Seq, self).__init__()
        self.bias = bias
        self.w_init = w_init
        self.hidden_size = hidden_size

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=self.hidden_size,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size, bias = self.bias, w_init=self.w_init)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=self.hidden_size)
        )

        self.sequential.add_module(
            f"convlstm2", ConvLSTM(
                in_channels=self.hidden_size, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size, bias = self.bias, w_init=self.w_init)
            )
            
        self.sequential.add_module(
            f"batchnorm2", nn.BatchNorm3d(num_features=num_kernels)
            )

        # Add rest of the layers
        for l in range(3, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size, bias = self.bias, w_init=self.w_init)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                )
                
        # Add Convolutional Layer to predict output frame
        self.conv1 = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_kernels,
            kernel_size=kernel_size, padding=padding, bias = self.bias)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(
            in_channels=num_kernels, out_channels=32,
            kernel_size=kernel_size, padding=padding, bias = self.bias)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding, bias = self.bias)
        
        self.decoder = nn.Sequential(
            self.conv1,
            self.activation,
            self.dropout1,
            self.conv2,
            self.activation,
            self.dropout2,
            self.conv3
        )

        self.fc1 = nn.Linear(num_kernels * frame_size[0] * frame_size[1], 1024, bias = self.bias)
        self.dropout3 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(1024, 512, bias = self.bias)
        self.batch1 = nn.BatchNorm1d(num_features = 512)
        self.fc3 = nn.Linear(512, 128, bias = self.bias)
        self.dropout4 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            self.fc1,
            self.activation,
            self.dropout3,
            self.fc2,
            self.activation,
            self.batch1,
            self.fc3,
            self.activation,
            self.dropout4,
            self.fc4
        )        
       
        if self.w_init == True:
            self.apply(self.weights_init)
    
    
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

        # Forward propagation through all the layers
        output = self.sequential(X)
        recon_frame = self.decoder(output[:,:,-1])
        
        # Return only the last output frasme
        temp = output[:,:,-1]
        batch_size, n_channels, height, width = temp.size()
        flatten_size = n_channels * height * width
        flatten_frame = temp.reshape(batch_size, flatten_size)
        out = self.fc(flatten_frame)

        return out, recon_frame
