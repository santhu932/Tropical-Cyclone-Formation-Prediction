import numpy as np
import torch
import math
import random
import pandas as pd
import torchvision.transforms as transforms


def data_preprocessing(data):
    random.seed(10)
    dataframe = pd.read_csv('tc_0h.csv')
    
    dataframe['TC'] = dataframe['TC'].astype(int)
    dataframe['Is Other TC Happening'] = dataframe['Is Other TC Happening'].astype(int)
    tc_labels = dataframe['TC']
    other_tcLabels = dataframe['Is Other TC Happening']
    
    other_negative_indices = [index for index, value in enumerate(other_tcLabels) if value == 0]
    other_positive_indices = [index for index, value in enumerate(other_tcLabels) if value == 1]
    
    positive_indices = [index for index, value in enumerate(tc_labels) if value == 1]
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
    data = data.numpy()
    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(indices),4,data.shape[1],data.shape[2], data.shape[3]))
    for i , val in enumerate(indices):
        sampled_data[i,0,:,:,: ] = reshaped_data[val - 3, 0, :, :,:]
        sampled_data[i,1,:,:,: ] = reshaped_data[val - 2, 0, :, :,:]
        sampled_data[i,2,:,:,: ] = reshaped_data[val - 1, 0, :, :,:]
        sampled_data[i,3,:,:,: ] = reshaped_data[val - 0, 0, :, :,:]
    
    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        #transforms.RandomCrop((41 - 10, 161 - 10)),
        transforms.ToTensor()
    ])
    sampled_data = torch.from_numpy(sampled_data)
    augmented_data = torch.zeros(len(positive_indices), sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4])
    j = 0
    for i, sample in enumerate(sampled_data):
        if labels[i] == 1:
            temp = torch.zeros(sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3])
            for t in range(4):
                temp1 = torch.zeros(sample.shape[1], sample.shape[2], sample.shape[3])
                for k, image in enumerate(sample[t]):
                    temp1[k] = transform1(image)
                temp[t] = temp1
            augmented_data[j] = temp
            j += 1
            
    augmented_data = augmented_data.numpy()
    sampled_data = sampled_data.numpy()
    
    total_data = []
    for i in range(sampled_data.shape[0]):
        total_data.append((sampled_data[i], labels[i]))
        
    for i in range(augmented_data.shape[0]):
        total_data.append((augmented_data[i], 1))
    random.shuffle(total_data)
    
    final_data = np.zeros((sampled_data.shape[0] + augmented_data.shape[0], sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    target = []
    for i, pair in enumerate(total_data):
        final_data[i] = pair[0]
        target.append(int(pair[1]))
    target = np.array(target)
    train_data = final_data[:int(0.8 * len(final_data))]
    train_target = target[:int(0.8 * len(final_data))]
    train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
    train_data = train_data.permute(0,2,1,3,4)
    
    test_data = final_data[int(0.8 * len(final_data)):]
    test_target = target[int(0.8 * len(final_data)):]
    test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)
    test_data = test_data.permute(0,2,1,3,4)
    
    return train_data, train_target, test_data, test_target


def data_preprocessing_forecasting(data, config):
    random.seed(config.s)
    dataframe = pd.read_csv('tc_0h.csv')
    
    dataframe['TC'] = dataframe['TC'].astype(int)
    dataframe['Is Other TC Happening'] = dataframe['Is Other TC Happening'].astype(int)
    tc_labels = dataframe['TC']
    other_tcLabels = dataframe['Is Other TC Happening']
    
    other_negative_indices = [index for index, value in enumerate(other_tcLabels) if value == 0]
    other_positive_indices = [index for index, value in enumerate(other_tcLabels) if value == 1]
    
    positive_indices = [index for index, value in enumerate(tc_labels) if value == 1]
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
    data = data.numpy()
    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(indices),4,data.shape[1],data.shape[2], data.shape[3]))
    target_frame = np.zeros((len(indices), 2, data.shape[1],data.shape[2], data.shape[3]))
    for i , val in enumerate(indices):
        sampled_data[i,0, :, :, :] = reshaped_data[val - 5, 0, :, :, :]
        sampled_data[i,1, :, :, :] = reshaped_data[val - 4, 0, :, :, :]
        sampled_data[i,2, :, :, :] = reshaped_data[val - 3, 0, :, :, :]
        sampled_data[i,3, :, :, :] = reshaped_data[val - 2, 0, :, :, :]
        target_frame[i,0, :, :, :] = reshaped_data[val - 1, 0, :, :, :]
        target_frame[i,1, :, :, :] = reshaped_data[val, 0, :, :, :]
    
    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        #transforms.RandomCrop((41 - 10, 161 - 10)),
        transforms.ToTensor()
    ])
    sampled_data = torch.from_numpy(sampled_data)
    target_frame = torch.from_numpy(target_frame)
    augmented_data = torch.zeros(len(positive_indices), sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4])
    agumented_target_frame = torch.zeros(len(positive_indices), 2, sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4])
    j = 0
    for i, sample in enumerate(sampled_data):
        if labels[i] == 1:
            agumented_target_frame[j] = target_frame[i]
            temp = torch.zeros(sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3])
            for t in range(4):
                temp1 = torch.zeros(sample.shape[1], sample.shape[2], sample.shape[3])
                for k, tensor in enumerate(sample[t]):
                    temp1[k] = transform1(tensor)
                temp[t] = temp1
            #augmented_tensor = torch.stack([transform1(tensor) for tensor in sample])
            augmented_data[j] = temp
            j += 1
            
    augmented_data = augmented_data.numpy()
    sampled_data = sampled_data.numpy()
    agumented_target_frame = agumented_target_frame.numpy()
    
    total_data = []
    for i in range(sampled_data.shape[0]):
        total_data.append((sampled_data[i], labels[i], target_frame[i]))
        
    for i in range(augmented_data.shape[0]):
        total_data.append((augmented_data[i], 1, agumented_target_frame[i]))
    random.shuffle(total_data)
    
    
    final_data = np.zeros((sampled_data.shape[0] + augmented_data.shape[0], sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    target = []
    final_target_frame = np.zeros((sampled_data.shape[0] + augmented_data.shape[0], 2, sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    
    
    del augmented_data
    del sampled_data
    del agumented_target_frame
    del target_frame
    del reshaped_data
    del indices
    del positive_indices
    del positive_otherTc_indices
    del sampled_positive_otherTc_indices
    del other_tcLabels
    del other_negative_indices
    del other_positive_indices
    del negative_indices
    del negative_tc_indices
    del sampled_negative_tc_indices
    
    for i, pair in enumerate(total_data):
        final_data[i] = pair[0]
        target.append(int(pair[1]))
        final_target_frame[i] = pair[2]
        
    del total_data
    
    target = np.array(target)
    train_data = final_data[:int(0.8 * len(final_data))]
    train_target = target[:int(0.8 * len(final_data))]
    train_output_frame = final_target_frame[:int(0.8 * len(final_data))]
    train_data, train_target, train_output_frame = torch.from_numpy(train_data), torch.from_numpy(train_target), torch.from_numpy(train_output_frame)
    train_data = train_data.permute(0,2,1,3,4)
    
    test_data = final_data[int(0.8 * len(final_data)):]
    test_target = target[int(0.8 * len(final_data)):]
    test_output_frame = final_target_frame[int(0.8 * len(final_data)):]
    test_data, test_target, test_output_frame = torch.from_numpy(test_data), torch.from_numpy(test_target), torch.from_numpy(test_output_frame)
    test_data = test_data.permute(0,2,1,3,4)
    
    del final_data
    del final_target_frame
    del target
    
    return train_data, train_target, train_output_frame, test_data, test_target, test_output_frame 


def data_preprocessing_forecasting_old(data, t):

    data = data.numpy()
    labels = t[5:]

    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(labels),4,data.shape[1],data.shape[2], data.shape[3]))
    target_frame = np.zeros((len(labels), 2, data.shape[1],data.shape[2], data.shape[3]))
    for i in range(len(labels)):
        sampled_data[i,0, :, :, :] = reshaped_data[i + 0, 0, :, :, :]
        sampled_data[i,1, :, :, :] = reshaped_data[i + 1, 0, :, :, :]
        sampled_data[i,2, :, :, :] = reshaped_data[i + 2, 0, :, :, :]
        sampled_data[i,3, :, :, :] = reshaped_data[i + 3, 0, :, :, :]
        target_frame[i,0, :, :, :] = reshaped_data[i + 4, 0, :, :, :]
        target_frame[i,1, :, :, :] = reshaped_data[i + 5, :, :, :]
    
    labels = np.array(labels)
    train_data = sampled_data[:int(0.8 * len(sampled_data))]
    train_target = labels[:int(0.8 * len(sampled_data))]
    train_output_frame = target_frame[:int(0.8 * len(sampled_data))]
    train_data, train_target, train_output_frame = torch.from_numpy(train_data), torch.from_numpy(train_target), torch.from_numpy(train_output_frame)
    train_data = train_data.permute(0,2,1,3,4)
    
    test_data = sampled_data[int(0.8 * len(sampled_data)):]
    test_target = labels[int(0.8 * len(sampled_data)):]
    test_output_frame = target_frame[int(0.8 * len(sampled_data)):]
    test_data, test_target, test_output_frame = torch.from_numpy(test_data), torch.from_numpy(test_target), torch.from_numpy(test_output_frame)
    test_data = test_data.permute(0,2,1,3,4)
    
    del sampled_data
    del labels
    del target_frame
    
    return train_data, train_target, train_output_frame, test_data, test_target, test_output_frame

def data_preprocessing_old(data, t):
    
    data = data.numpy()
    labels = t[3:]

    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(labels),4,data.shape[1],data.shape[2], data.shape[3]))
    for i in range(len(labels)):
        sampled_data[i,0, :, :, :] = reshaped_data[i + 0, 0, :, :, :]
        sampled_data[i,1, :, :, :] = reshaped_data[i + 1, 0, :, :, :]
        sampled_data[i,2, :, :, :] = reshaped_data[i + 2, 0, :, :, :]
        sampled_data[i,3, :, :, :] = reshaped_data[i + 3, 0, :, :, :]
    
    labels = np.array(labels)
    train_data = sampled_data[:int(0.8 * len(sampled_data))]
    train_target = labels[:int(0.8 * len(sampled_data))]
    train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
    train_data = train_data.permute(0,2,1,3,4)
    
    test_data = sampled_data[int(0.8 * len(sampled_data)):]
    test_target = labels[int(0.8 * len(sampled_data)):]
    test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)
    test_data = test_data.permute(0,2,1,3,4)

    del sampled_data
    del labels

    return train_data, train_target, test_data, test_target


def data_preprocessing_new(data, config):
    random.seed(config.s)
    dataframe = pd.read_csv('tc_0h.csv')
    
    dataframe['TC'] = dataframe['TC'].astype(int)
    dataframe['Is Other TC Happening'] = dataframe['Is Other TC Happening'].astype(int)
    tc_labels = dataframe['TC']
    other_tcLabels = dataframe['Is Other TC Happening']
    
    other_negative_indices = [index for index, value in enumerate(other_tcLabels) if value == 0]
    other_positive_indices = [index for index, value in enumerate(other_tcLabels) if value == 1]
    
    positive_indices = [index for index, value in enumerate(tc_labels) if value == 1]
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
    data = data.numpy()
    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(indices),4,data.shape[1],data.shape[2], data.shape[3]))
    for i , val in enumerate(indices):
        sampled_data[i,0,:,:,: ] = reshaped_data[val - 3, 0, :, :,:]
        sampled_data[i,1,:,:,: ] = reshaped_data[val - 2, 0, :, :,:]
        sampled_data[i,2,:,:,: ] = reshaped_data[val - 1, 0, :, :,:]
        sampled_data[i,3,:,:,: ] = reshaped_data[val - 0, 0, :, :,:]
    
    transform1 = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        #transforms.RandomCrop((41 - 10, 161 - 10)),
        #transforms.ToTensor()
    ])
    sampled_data = torch.from_numpy(sampled_data)
    augmented_data = torch.zeros(len(positive_indices), sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4])
    j = 0
    for i, sample in enumerate(sampled_data):
        if labels[i] == 1:
            #augmented_tensor = torch.stack([transform1(tensor) for tensor in sample])
            #temp = torch.zeros(sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3])
            # for t in range(4):
            #     temp1 = torch.zeros(sample.shape[1], sample.shape[2], sample.shape[3])
            #     for k, image in enumerate(sample[t]):
            #         temp1[k] = transform1(image)
            #     temp[t] = temp1
            augmented_data[j] = transform1(sample)
            j += 1
            
    augmented_data = augmented_data.numpy()
    sampled_data = sampled_data.numpy()
    
    total_data = []
    for i in range(sampled_data.shape[0]):
        total_data.append((sampled_data[i], labels[i]))
        
    for i in range(augmented_data.shape[0]):
        total_data.append((augmented_data[i], 1))
    random.shuffle(total_data)
    
    final_data = np.zeros((sampled_data.shape[0] + augmented_data.shape[0], sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    target = []
    for i, pair in enumerate(total_data):
        final_data[i] = pair[0]
        target.append(int(pair[1]))
    target = np.array(target)
    train_data = final_data[:int(0.8 * len(final_data))]
    train_target = target[:int(0.8 * len(final_data))]
    train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
    train_data = train_data.permute(0,2,1,3,4)
    
    test_data = final_data[int(0.8 * len(final_data)):]
    test_target = target[int(0.8 * len(final_data)):]
    test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)
    test_data = test_data.permute(0,2,1,3,4)
    
    return train_data, train_target, test_data, test_target


def data_preprocessing_forecasting_new(data, config):
    random.seed(config.s)
    dataframe = pd.read_csv('tc_0h.csv')
    
    dataframe['TC'] = dataframe['TC'].astype(int)
    dataframe['Is Other TC Happening'] = dataframe['Is Other TC Happening'].astype(int)
    tc_labels = dataframe['TC']
    other_tcLabels = dataframe['Is Other TC Happening']
    
    other_negative_indices = [index for index, value in enumerate(other_tcLabels) if value == 0]
    other_positive_indices = [index for index, value in enumerate(other_tcLabels) if value == 1]
    
    positive_indices = [index for index, value in enumerate(tc_labels) if value == 1]
    positive_indices = positive_indices[0:1] + positive_indices[2:]
    #print(positive_indices[1])
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
    data = data.numpy()
    reshaped_data = data.reshape(data.shape[0],1, data.shape[1],data.shape[2], data.shape[3])
    sampled_data = np.zeros((len(indices),6,data.shape[1],data.shape[2], data.shape[3]))
    
#    example_data = np.zeros((1,4,data.shape[1],data.shape[2], data.shape[3]))
#    example_target = np.zeros((1,2,data.shape[1],data.shape[2], data.shape[3]))
#    example_data[0,0, :, :, :] = reshaped_data[46, 0, :, :, :]
#    example_data[0,1, :, :, :] = reshaped_data[47, 0, :, :, :]
#    example_data[0,2, :, :, :] = reshaped_data[48, 0, :, :, :]
#    example_data[0,3, :, :, :] = reshaped_data[49, 0, :, :, :]
#    example_target[0,0, :, :, :] = reshaped_data[50, 0, :, :, :]
#    example_target[0,1, :, :, :] = reshaped_data[51, 0, :, :, :]
    
    for i , val in enumerate(indices):
        sampled_data[i,0, :, :, :] = reshaped_data[val - 5, 0, :, :, :]
        sampled_data[i,1, :, :, :] = reshaped_data[val - 4, 0, :, :, :]
        sampled_data[i,2, :, :, :] = reshaped_data[val - 3, 0, :, :, :]
        sampled_data[i,3, :, :, :] = reshaped_data[val - 2, 0, :, :, :]
        sampled_data[i,4, :, :, :] = reshaped_data[val - 1, 0, :, :, :]
        sampled_data[i,5, :, :, :] = reshaped_data[val - 0, 0, :, :, :]
    
    transform1 = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        #transforms.RandomCrop((41 - 10, 161 - 10)),
        #transforms.ToTensor()
    ])

    sampled_data = torch.from_numpy(sampled_data)
    augmented_data = torch.zeros(len(positive_indices), sampled_data.shape[1], sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4])

    j = 0
    for i, sample in enumerate(sampled_data):
        if labels[i] == 1:
            #augmented_tensor = torch.stack([transform1(tensor) for tensor in sample])
            augmented_data[j] = transform1(sample)
            j += 1
            
    augmented_data = augmented_data.numpy()
    sampled_data = sampled_data.numpy()

    sampled_input_data = np.zeros((sampled_data.shape[0], 4, sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    sampled_target_data = np.zeros((sampled_data.shape[0], 2, sampled_data.shape[2], sampled_data.shape[3], sampled_data.shape[4]))
    
    for i in range(sampled_data.shape[0]):
        sampled_input_data[i,0, :, :, :] = sampled_data[i,0, :, :, :]
        sampled_input_data[i,1, :, :, :] = sampled_data[i,1, :, :, :]
        sampled_input_data[i,2, :, :, :] = sampled_data[i,2, :, :, :]
        sampled_input_data[i,3, :, :, :] = sampled_data[i,3, :, :, :]
        sampled_target_data[i,0, :, :, :] = sampled_data[i,4, :, :, :]
        sampled_target_data[i,1, :, :, :] = sampled_data[i,5, :, :, :]
    del sampled_data

    augmented_input_data = np.zeros((augmented_data.shape[0], 4, augmented_data.shape[2], augmented_data.shape[3], augmented_data.shape[4]))
    augmented_target_data = np.zeros((augmented_data.shape[0], 2, augmented_data.shape[2], augmented_data.shape[3], augmented_data.shape[4]))
    for i in range(augmented_data.shape[0]):
        augmented_input_data[i,0, :, :, :] = augmented_data[i,0, :, :, :]
        augmented_input_data[i,1, :, :, :] = augmented_data[i,1, :, :, :]
        augmented_input_data[i,2, :, :, :] = augmented_data[i,2, :, :, :]
        augmented_input_data[i,3, :, :, :] = augmented_data[i,3, :, :, :]
        augmented_target_data[i,0, :, :, :] = augmented_data[i,4, :, :, :]
        augmented_target_data[i,1, :, :, :] = augmented_data[i,5, :, :, :]
    del augmented_data

    total_data = []
    for i in range(sampled_input_data.shape[0]):
        total_data.append((sampled_input_data[i], labels[i], sampled_target_data[i]))
        
    for i in range(augmented_input_data.shape[0]):
        total_data.append((augmented_input_data[i], 1, augmented_target_data[i]))
    random.shuffle(total_data)
    
    
    final_data = np.zeros((sampled_input_data.shape[0] + augmented_input_data.shape[0], sampled_input_data.shape[1], sampled_input_data.shape[2], sampled_input_data.shape[3], sampled_input_data.shape[4]))
    target = []
    final_target_frame = np.zeros((sampled_input_data.shape[0] + augmented_input_data.shape[0], sampled_target_data.shape[1], sampled_target_data.shape[2], sampled_target_data.shape[3], sampled_target_data.shape[4]))
    
    del sampled_input_data
    del sampled_target_data
    del augmented_input_data
    del augmented_target_data
    del reshaped_data
    del indices
    del positive_indices
    del positive_otherTc_indices
    del sampled_positive_otherTc_indices
    del other_tcLabels
    del other_negative_indices
    del other_positive_indices
    del negative_indices
    del negative_tc_indices
    del sampled_negative_tc_indices
    
    for i, pair in enumerate(total_data):
        final_data[i] = pair[0]
        target.append(int(pair[1]))
        final_target_frame[i] = pair[2]
        
    del total_data
    
    target = np.array(target)
    train_data = final_data[:int(0.8 * len(final_data))]
    train_target = target[:int(0.8 * len(final_data))]
    train_output_frame = final_target_frame[:int(0.8 * len(final_data))]
    train_data, train_target, train_output_frame = torch.from_numpy(train_data), torch.from_numpy(train_target), torch.from_numpy(train_output_frame)
    train_data = train_data.permute(0,2,1,3,4)
    
#    example_data, example_target = torch.from_numpy(example_data), torch.from_numpy(example_target)
#    example_data = example_data.permute(0,2,1,3,4)
    
    test_data = final_data[int(0.8 * len(final_data)):]
    test_target = target[int(0.8 * len(final_data)):]
    test_output_frame = final_target_frame[int(0.8 * len(final_data)):]
    test_data, test_target, test_output_frame = torch.from_numpy(test_data), torch.from_numpy(test_target), torch.from_numpy(test_output_frame)
    test_data = test_data.permute(0,2,1,3,4)
    
    del final_data
    del final_target_frame
    del target
    
    return train_data, train_target, train_output_frame, test_data, test_target, test_output_frame
