import numpy as np
import torch
import math
import random
import pandas as pd
import torchvision.transforms as transforms

def get_indices_labels(filepath):

    dataframe = pd.read_csv(filepath)
    dataframe['TC'] = dataframe['TC'].astype(int)
    dataframe['Is Other TC Happening'] = dataframe['Is Other TC Happening'].astype(int)
    tc_labels = dataframe['TC']
    other_tcLabels = dataframe['Is Other TC Happening']

    other_negative_indices = [index for index, value in enumerate(other_tcLabels) if value == 0]
    other_positive_indices = [index for index, value in enumerate(other_tcLabels) if value == 1]

    positive_indices = [index for index, value in enumerate(tc_labels) if value == 1]
    #positive_indices = positive_indices[0:1] + positive_indices[2:]
    negative_tc_indices = [index for index in other_negative_indices if index not in positive_indices]
    positive_otherTc_indices = [index for index in other_positive_indices if index not in positive_indices]

    sampled_negative_tc_indices = random.sample(negative_tc_indices, k=int(len(negative_tc_indices) * 0.16))
    sampled_positive_otherTc_indices = random.sample(positive_otherTc_indices, k=int(len(positive_otherTc_indices) * 0.07))
    negative_indices = sampled_negative_tc_indices + sampled_positive_otherTc_indices
    negative_indices.sort()
    cut_negative_indices = negative_indices[4:]
    indices = positive_indices + cut_negative_indices
    indices.sort()
    
    labels = np.zeros((len(indices)), dtype = int)
    for i, ind in enumerate(indices):
        if ind in positive_indices:
            labels[i] = 1
        else:
            labels[i] = 0
    
    return indices, labels, len(positive_indices)

def get_spatial_locations(filepath):
    dataframe = pd.read_csv(filepath)
    offset = (5, 100)
    dataframe['Latitude'] = dataframe['Latitude'].fillna(0).astype(int)
    dataframe['Longitude'] = dataframe['Longitude'].fillna(0).astype(int)
    latitudes = dataframe['Latitude'] - offset[0]
    longitudes = dataframe['Longitude'] - offset[1]
    return latitudes, longitudes

def get_gaussian_probs(latitude, longitude):
    gaussian_radius = 3
    clip_prob = 0.1
    x = np.arange(0, 41, 1)
    y = np.arange(0, 161, 1)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    gaussian_probs = np.exp(-((xx - latitude) ** 2 + (yy - longitude) ** 2) / (2 * gaussian_radius ** 2))
    gaussian_probs[gaussian_probs < clip_prob] = 0
    return gaussian_probs
    

def load_data(data, config):
    random.seed(config.s)
    filepath = 'tc_0h.csv'
    indices, labels, num_positive_indices = get_indices_labels(filepath) #get sampled indices and their binary labels
    latitudes, longitudes = get_spatial_locations(filepath) #get spatial loaction of tc formation
    data = data.numpy()
    reshaped_data = data.reshape(data.shape[0], 1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(indices), config.num_timesteps + config.future_interval + 1,data.shape[1],data.shape[2], data.shape[3]))

    #Creating single 5D numpy array storing labels and input data for specified timesteps and future forecasting interval for data augmentation
    for i , val in enumerate(indices):
        k = 0
        while k < sampled_data.shape[1] - 1:
            sampled_data[i, k, :, :, :] = reshaped_data[val - (sampled_data.shape[1] - 2) - k, 0, :, :, :]
            k += 1
        if labels[i] == 1:
            label_grid  = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
            gaussian_probs = get_gaussian_probs(latitudes[val], longitudes[val])
            label_grid = label_grid + gaussian_probs
            sampled_data[i, k, :, :, :] = label_grid
        else:
            label_zeros = np.zeros((data.shape[1], data.shape[2], data.shape[3]))
            sampled_data[i, k, :, :, :] = label_zeros
    
    
    transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
    ])
    
    #Data augmentation for positive indices
    sampled_data = torch.from_numpy(sampled_data)
    augmented_data = torch.zeros(num_positive_indices, sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4])
    j = 0
    for i, sample in enumerate(sampled_data):
        if labels[i] == 1:
            augmented_data[j] = transform1(sample)
            j += 1

    augmented_data = augmented_data.numpy()
    sampled_data = sampled_data.numpy()

    #Splitting the original data into input data, target frames, spatial grid labels
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
    
    #Deleting arrays which are no longer needed
    del sampled_data
    del longitudes
    del latitudes

    #Splitting the augmented data into input data, target frames, spatial grid labels
    augmented_input_data = np.zeros((augmented_data.shape[0], config.num_timesteps, augmented_data.shape[2], augmented_data.shape[3], augmented_data.shape[4]))
    augmented_target_data = np.zeros((augmented_data.shape[0], config.future_interval, augmented_data.shape[2], augmented_data.shape[3], augmented_data.shape[4]))
    augmented_grid_label = np.zeros((augmented_data.shape[0], 1, augmented_data.shape[3], augmented_data.shape[4]))
    
    for i in range(augmented_data.shape[0]):
        k = 0
        while k < config.num_timesteps:
            augmented_input_data[i, k, :, :, :] = augmented_data[i, k, :, :, :]
            k += 1
        j = 0
        while k < config.num_timesteps + config.future_interval:
            augmented_target_data[i, j, :, :, :] = augmented_data[i, k, :, :, :]
            k += 1
            j += 1
        augmented_grid_label[i, 0, :, :] = augmented_data[i, k, 0, :, :]
        
    del augmented_data

    #Merging both original and augmented data into one single array
    total_data = []
    for i in range(sampled_input_data.shape[0]):
        total_data.append((sampled_input_data[i], labels[i], sampled_target_data[i], sampled_grid_label[i]))

    for i in range(augmented_input_data.shape[0]):
        total_data.append((augmented_input_data[i], 1, augmented_target_data[i], augmented_grid_label[i]))
        
    random.shuffle(total_data)

    final_data = np.zeros((sampled_input_data.shape[0] + augmented_input_data.shape[0], sampled_input_data.shape[1], sampled_input_data.shape[2], sampled_input_data.shape[3], sampled_input_data.shape[4]))
    target = []
    final_target_frame = np.zeros((sampled_input_data.shape[0] + augmented_input_data.shape[0], sampled_target_data.shape[1], sampled_target_data.shape[2], sampled_target_data.shape[3], sampled_target_data.shape[4]))
    final_grid_label = np.zeros((sampled_input_data.shape[0] + augmented_input_data.shape[0], sampled_grid_label.shape[1], sampled_grid_label.shape[2], sampled_grid_label.shape[3]))

    del sampled_input_data
    del sampled_target_data
    del sampled_grid_label
    del augmented_grid_label
    del augmented_input_data
    del augmented_target_data
    del reshaped_data
    del indices
    
    for i, pair in enumerate(total_data):
        final_data[i] = pair[0]
        target.append(int(pair[1]))
        final_target_frame[i] = pair[2]
        final_grid_label[i] = pair[3]
        
    del total_data

    target = np.array(target)
    train_data = final_data[:int(0.8 * len(final_data))]
    train_target = target[:int(0.8 * len(final_data))]
    train_output_frame = final_target_frame[:int(0.8 * len(final_data))]
    train_grid_labels = final_grid_label[:int(0.8 * len(final_data))]
    train_data, train_target, train_output_frame, train_grid_labels = torch.from_numpy(train_data), torch.from_numpy(train_target), torch.from_numpy(train_output_frame), torch.from_numpy(train_grid_labels)
    train_data = train_data.permute(0,2,1,3,4)


    test_data = final_data[int(0.8 * len(final_data)):]
    test_target = target[int(0.8 * len(final_data)):]
    test_output_frame = final_target_frame[int(0.8 * len(final_data)):]
    test_grid_labels = final_grid_label[int(0.8 * len(final_data)):]
    test_data, test_target, test_output_frame, test_grid_labels = torch.from_numpy(test_data), torch.from_numpy(test_target), torch.from_numpy(test_output_frame), torch.from_numpy(test_grid_labels)
    test_data = test_data.permute(0,2,1,3,4)

    del final_data
    del final_target_frame
    del target
    del final_grid_label

    return train_data, train_target, train_output_frame, train_grid_labels, test_data, test_target, test_output_frame, test_grid_labels
