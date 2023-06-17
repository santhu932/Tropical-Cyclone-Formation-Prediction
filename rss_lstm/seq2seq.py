import torch.nn as nn
import torch
from rss_lstm.convLSTM import ConvLSTM
from vit_pytorch import ViT
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
    activation, frame_size, num_layers, num_timesteps, bias, w_init, hidden_size):

        super(Seq2Seq, self).__init__()
        self.bias = bias
        self.w_init = w_init
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "leaky_relu":
            self.activation = nn.functional.leaky_relu
        elif activation == "elu":
            self.activation = nn.ELU()

        #self.sequential = nn.Sequential()
        self.module_list = nn.ModuleList()

        # Add First layer (Different in_channels than the rest)
        self.module_list.append(ConvLSTM(
                in_channels=num_channels, out_channels=self.hidden_size,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size, bias = self.bias, w_init=self.w_init)),
        self.module_list.append(nn.BatchNorm2d(num_features=self.hidden_size))


        self.module_list.append(ConvLSTM(
                in_channels=self.hidden_size, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size, bias = self.bias, w_init=self.w_init)),
        self.module_list.append(nn.BatchNorm2d(num_features=num_kernels))


        # Add rest of the layers
        for l in range(3, num_layers+1):

            self.module_list.append(ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size, bias = self.bias, w_init=self.w_init))
            self.module_list.append(nn.BatchNorm2d(num_features=num_kernels))

                
        # Add Convolutional Layer to predict output frame
        self.conv1 = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_kernels,
            kernel_size=kernel_size, padding=padding, bias = self.bias)
        self.batchnorm1 = nn.BatchNorm2d(num_features=num_kernels)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(
            in_channels=num_kernels, out_channels=32,
            kernel_size=kernel_size, padding=padding, bias = self.bias)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding, bias = self.bias)
        
        self.decoder = nn.Sequential(
            self.conv1,
            self.activation,
            self.batchnorm1,
            self.dropout1,
            self.conv2,
            self.activation,
            self.batchnorm2,
            self.dropout2,
            self.conv3
        )

        self.fc1 = nn.Linear(num_kernels * frame_size[0] * frame_size[1], 1024, bias = self.bias)
        self.batchnorm3 = nn.BatchNorm1d(num_features=1024)
        self.dropout3 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(1024, 512, bias = self.bias)
        self.batchnorm4 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(512, 128, bias = self.bias)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(128, 1)
        

        self.fc = nn.Sequential(
            self.fc1,
            self.activation,
            self.batchnorm3,
            self.dropout3,
            self.fc2,
            self.activation,
            self.batchnorm4,
            self.fc3,
            self.activation,
            self.batchnorm5,
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
			

		
                
    def forward(self, X, target_frames = None, prob_mask = None, prob_mask1 = None, prediction = False):
        previous_H = {}
        previous_C = {}
        for t in range(self.num_timesteps):
            if t == 0:
                x = X[:,:,t]
                for i, module in enumerate(self.module_list):
                    if i % 2 == 0:
                        name = f"convlstm{(i // 2) + 1}"
                        output, C = module(x)
                        previous_H[name] = output
                        previous_C[name] = C
                    else:
                        output = module(x)
                    x = output
                recon_frame = self.decoder(x)
            else:
                #x = X[:,:,t]
                if prediction == False:
                    x = prob_mask[:,:,t-1] * X[:,:,t] + (1 - prob_mask[: , :, t-1]) * recon_frame
                else:
                    x = X[:,:,t]
                for i, module in enumerate(self.module_list):
                    if i % 2 == 0:
                        name = f"convlstm{(i // 2) + 1}"
                        output, C = module(x, previous_H[name], previous_C[name])
                        previous_H[name] = output
                        previous_C[name] = C
                    else:
                        output = module(x)
                    x = output
                recon_frame = self.decoder(x)

        
        #temp = output.clone()
        recon_frames = torch.zeros(recon_frame.shape[0], 2, recon_frame.shape[1], recon_frame.shape[2], recon_frame.shape[3], device = device)
        recon_frames[:,0] = recon_frame.clone()
        #recon_frame = recon_frame.reshape(recon_frame.shape[0], recon_frame.shape[1], recon_frame.shape[2], recon_frame.shape[3])
        if prediction == False:
            recon_frame = prob_mask1[:,:,0] * recon_frame + (1 - prob_mask1[: , :, 0]) * target_frames[:, 0]
        for i, module in enumerate(self.module_list):
            if i % 2 == 0:
                name = f"convlstm{3 - (i//2)}"
                output, C = module(recon_frame, previous_H[name], previous_C[name])
                previous_H[name] = output
                previous_C[name] = C
            else:
                output = module(recon_frame)
            recon_frame = output
        recon_frame = self.decoder(recon_frame)
        recon_frames[:,1] = recon_frame.clone()

        
        #output = output - temp
        batch_size, n_channels, height, width = output.size()
        flatten_size = n_channels * height * width
        flatten_frame = output.reshape(batch_size, flatten_size)
        out = self.fc(flatten_frame)
        
        return out, recon_frames
    
