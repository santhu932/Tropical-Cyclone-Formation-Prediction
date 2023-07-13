import numpy as np
import torch
import math
import random
import pandas as pd
import torchvision.transforms as transforms

def load_data1(data, config):
    random.seed(config.s)
    dataframe = pd.read_csv('tc_0h.csv')
    offset = (5, 100)

    dataframe['Is Other TC Happening'] = dataframe['Is Other TC Happening'].astype(int)
    tc_labels = dataframe['Is Other TC Happening']
    dataframe['Other TC Locations'] = dataframe['Other TC Locations'].fillna("[]").apply(lambda x: eval(x))
    locations  = dataframe['Other TC Locations']
    data = data.numpy()
    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((data.shape[0] - config.num_timesteps,config.num_timesteps + config.future_interval + 1,data.shape[1],data.shape[2], data.shape[3]))

    def get_gaussian_probs(latitude, longitude):
        gaussian_radius = 3
        clip_prob = 0.1
        x = np.arange(0, 41, 1)
        y = np.arange(0, 161, 1)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        gaussian_probs = np.exp(-((xx - latitude) ** 2 + (yy - longitude) ** 2) / (2 * gaussian_radius ** 2))
        gaussian_probs[gaussian_probs < clip_prob] = 0
        return gaussian_probs
    
    for i in range(config.num_timesteps, data.shape[0]):
        k = 0
        while k < sampled_data.shape[1] - 1:
            sampled_data[i - config.num_timesteps, k, :, :, :] = reshaped_data[i - (sampled_data.shape[1] - 2) - k, 0, :, :, :]
            k += 1
        if tc_labels[i] == 1:
            label_grid  = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
            for (lat,lon) in locations[i]:
                lat = lat - offset[0]
                lon = lon - offset[1]
                gaussian_probs = get_gaussian_probs(lat, lon)
                label_grid = label_grid + gaussian_probs
            sampled_data[i - config.num_timesteps, k, :, :, :] = label_grid
        else:
            label_zeros = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
            sampled_data[i - config.num_timesteps, k, :, :, :] = label_zeros
            
    del reshaped_data
    labels = tc_labels[config.num_timesteps:]
    print(len(labels))
    print(sampled_data.shape[0])
    
    train_sampled_data = sampled_data[:int(sampled_data.shape[0] * 0.8)]
    train_labels = labels[:int(sampled_data.shape[0] * 0.8)]
    train_data, train_target, train_output_frames, train_grid_labels = prepare_test_data(train_sampled_data, train_labels, config)
    
    test_sampled_data = sampled_data[int(sampled_data.shape[0] * 0.8):]
    test_labels = labels[int(sampled_data.shape[0] * 0.8):]
    test_data, test_target, test_output_frames, test_grid_labels = prepare_test_data(test_sampled_data, test_labels, config)

    return train_data, train_target, train_output_frames, train_grid_labels, test_data, test_target, test_output_frames, test_grid_labels

def prepare_test_data(sampled_data, labels, config):
    num_positive_indices = np.sum(labels)
    print("test labels")
    print(len(labels))
    print(num_positive_indices)
    
    #sampled_data = sampled_data.numpy()
    sampled_input_data = np.zeros((sampled_data.shape[0], config.num_timesteps, sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    sampled_target_data = np.zeros((sampled_data.shape[0], config.future_interval, sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    sampled_grid_label = np.zeros((sampled_data.shape[0], 1, sampled_data.shape[3], sampled_data.shape[4]))
    
    for i in range(sampled_data.shape[0]):
        k = 0
        while k < config.num_timesteps:
            sampled_input_data[i, k, :, :, :] = sampled_data[i, k, :, :, :]
            k += 1
        j = 0
        while k < config.num_timesteps + config.future_interval:
            sampled_target_data[i, j, :, :, :] = sampled_data[i, k, :, :, :]
            k += 1
            j += 1
        sampled_grid_label[i, 0, :, :] = sampled_data[i, k, 0, :, :]
        
    
    target = np.array(labels)
    final_data, target, final_target_frame, final_grid_label = torch.from_numpy(sampled_input_data), torch.from_numpy(target), torch.from_numpy(sampled_target_data), torch.from_numpy(sampled_grid_label)
    final_data = final_data.permute(0,2,1,3,4)
    
    return final_data, target, final_target_frame, final_grid_label
