import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from rss_lstm.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
import pandas as pd
import random
import math
from lstm import utils
import wandb

config = dict(epochs = 50,
				clip = 0.0, 
				s = 10, 
				num_channels = 15,
				num_kernels = 64,
				kernel_size = (5, 5),
				padding = (2, 2), 
				activation = "elu",
				frame_size = (41, 161), 
				num_layers=3, 
				bias = True,
				w_init = True,
				batch_size = 32,
				lr = 0.0005,
				hidden_size = 128,
				weight_decay = 0.0,
				num_timesteps = 4,
                threshold = 0.5,
                threshold_decay = 0.95,
                sampling_step1 = 15,
                sampling_step2 = 30
				)
              
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, output_frames):
        self.data = data
        self.labels = labels
        self.output_frames = output_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        output_frame = self.output_frames[idx]
        return sample, label, output_frame

    
    
def train_epoch(model, optimizer, train_loader, epoch, config):
    training_loss = 0.
    accs = []
    batch_count = 0
    examples_count = 0
    for i, (inputs, labels, target_output_frames) in enumerate(train_loader, 1):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        inputs, labels, target_output_frames = inputs.float().to(device), labels.float().to(device), target_output_frames.float().to(device)
        #RSS
        prob_mask = torch.rand(labels.shape[0], config.num_timesteps - 1)
        prob_mask1 = torch.rand(labels.shape[0], 1)
        prob_mask, prob_mask1 = prob_mask.to(device), prob_mask1.to(device)
        if epoch < config.sampling_step1:
            prob_mask = (prob_mask < config.threshold).float()
            prob_mask1 = (prob_mask1 < config.threshold).float()
        elif epoch < config.sampling_step2:
            prob_mask = (prob_mask < (1 - config.threshold * (config.threshold_decay ** (epoch - config.sampling_step1 + 1)))).float()
            prob_mask1 = (prob_mask1 < (1 - config.threshold * (config.threshold_decay ** (epoch - config.sampling_step1 + 1)))).float()
        else:
            prob_mask = (prob_mask < 1.0).float()
            prob_mask1 = (prob_mask1 < 1.0).float()
        prob_mask = prob_mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, config.num_channels, -1, config.frame_size[0], config.frame_size[1])
        prob_mask1 = prob_mask1.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, config.num_channels, -1, config.frame_size[0], config.frame_size[1])

        outputs, output_frames = model(inputs.float(), target_output_frames, prob_mask, prob_mask1)
        log_probs = torch.sigmoid(outputs)
        labels = labels.reshape(labels.shape[0], 1)
        loss_func = torch.nn.BCELoss(reduction='none')
        bce_loss = loss_func(log_probs, labels)
        bce_loss = bce_loss/torch.max(bce_loss)
        label_loss = torch.sum(bce_loss) / labels.shape[0]
        recon_loss = F.mse_loss(output_frames.float(), target_output_frames.float())
        recon_loss = recon_loss/config.batch_size
        loss = label_loss + recon_loss
        optimizer.zero_grad()
        loss.backward()
        if config.clip > 0:
            #torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            torch.nn.utils.clip_grad_value_(model.parameters(), config.clip)
        optimizer.step()

        examples_count += len(inputs)
        if ((i + 1) % 10) == 0:
            wandb.log({"epoch": epoch, "loss": loss}, step = examples_count)
        true_positive = torch.sum((log_probs > 0.5) * (labels == 1))
        true_negative = torch.sum((log_probs <= 0.5) * (labels == 0))
        false_positive = torch.sum((log_probs > 0.5) * (labels == 0))
        false_negative = torch.sum((log_probs <= 0.5) * (labels == 1))
        training_loss += loss.cpu().data.numpy()
        acc = (true_positive + true_negative)/(true_negative + true_positive + false_positive + false_negative)
        accs.append(acc.cpu().numpy())
    return training_loss / len(train_loader), np.mean(accs)

def eval_epoch(model, test_loader, epoch, config, example_data, example_target):
    test_loss = 0.
    all_predicted_labels = []
    all_labels = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for i, (inputs, labels, target_output_frames) in enumerate(test_loader, 1):
            example_data, example_target = example_data.float().to(device), example_target.float().to(device)
            inputs, labels, target_output_frames = inputs.float().to(device), labels.float().to(device), target_output_frames.to(device)
            if i == 1:
                inputs[0] = example_data[0]
                labels[0] = 1
                target_output_frames[0] = example_target[0]
            outputs, output_frames = model(inputs.float(), prediction = True)
            log_probs = torch.sigmoid(outputs)
            labels = labels.reshape(labels.shape[0], 1)
            loss_func = torch.nn.BCELoss(reduction='none')
            bce_loss = loss_func(log_probs, labels)
            bce_loss = bce_loss/torch.max(bce_loss)
            label_loss = torch.sum(bce_loss) / labels.shape[0]
            if epoch == 49 and i == 1:
                for j in range(labels.shape[0]):
                    if labels[j] == 1:
                        print(j)
                        target_array= target_output_frames.clone().detach().cpu().numpy()
                        predicted_array = output_frames.clone().detach().cpu().numpy()
                        target_arrayReshaped_tplus1 = target_array[j,0].reshape(target_array[j,0].shape[0], -1)
                        predicted_arrayReshaped_tplus1 = predicted_array[j,0].reshape(predicted_array[j,0].shape[0], -1)
                        target_arrayReshaped_tplus2 = target_array[j,1].reshape(target_array[j,1].shape[0], -1)
                        predicted_arrayReshaped_tplus2 = predicted_array[j,1].reshape(predicted_array[j,1].shape[0], -1)
                        np.savetxt('predicted_array_tplus1.txt', predicted_arrayReshaped_tplus1)
                        np.savetxt('target_array_tplus1.txt', target_arrayReshaped_tplus1)
                        np.savetxt('predicted_array_tplus2.txt', predicted_arrayReshaped_tplus2)
                        np.savetxt('target_array_tplus2.txt', target_arrayReshaped_tplus2)
                        break
                #torch.onnx.export(model, inputs, "model.onnx")
                #wandb.save("model.onnx")

            recon_loss = F.mse_loss(output_frames.float(), target_output_frames.float())
            recon_loss = recon_loss/config.batch_size
            loss =  label_loss + recon_loss
            true_positive += torch.sum((log_probs > 0.5) * (labels == 1))
            true_negative += torch.sum((log_probs <= 0.5) * (labels == 0))
            false_positive += torch.sum((log_probs > 0.5) * (labels == 0))
            false_negative += torch.sum((log_probs <= 0.5) * (labels == 1))
            test_loss += loss.cpu().data.numpy()
            all_predicted_labels.append(outputs.detach().cpu().numpy().argmax(axis=1))
            all_labels.append(labels.cpu().numpy())
    all_predicted_labels = np.concatenate(all_predicted_labels, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # accuracy
    acc = (true_positive + true_negative)/(true_negative + true_positive + false_positive + false_negative)
    stat = {'TP': true_positive, 'TN': true_negative, 'FP': false_positive, 'FN': false_negative}
    print(stat)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return test_loss / len(test_loader.dataset), acc, precision, recall, f1


def run_model(model, train_loader, test_loader, optimizer, config, example_data, example_target):
    wandb.watch(model, log="all", log_freq=10)
    epochs = config.epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch, config)
        test_loss, test_acc, precision, recall, f1 = eval_epoch(model, test_loader, epoch, config, example_data, example_target)
        loss_dict = {'train_loss': train_loss, 'test_loss': test_loss}
        wandb.log(loss_dict)
        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f}\tTest Loss: {:.4f}\tTest Accuracy: {:.4f}'.format(
                epoch, train_loss, train_acc, test_loss, test_acc))
        print('Precision: {:.4f} \tRecall: {:.4f}\tF1: {:.4f}'.format(precision, recall, f1))
     

def make(config):
    train_npz = np.load('ncep_WP_EP_new_2_binary.npz')
    data, target = train_npz['data'], train_npz['target']
    mean = np.mean(data, axis=(0, 2, 3))
    std = np.std(data, axis=(0, 2, 3))
    transform = transforms.Normalize(mean, std)
    data, target = torch.from_numpy(data), torch.from_numpy(target)
    data = transform(data)
    train_data, train_target, train_output_frame, test_data, test_target, test_output_frame, example_data, example_target = utils.data_preprocessing_forecasting_new(data, config)
    trainset = CustomDataset(train_data, train_target, train_output_frame)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size)
    
    testset = CustomDataset(test_data, test_target, test_output_frame)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size)

    model = Seq2Seq(num_channels=config.num_channels, num_kernels=config.num_kernels,
                kernel_size=config.kernel_size, padding=config.padding, activation=config.activation,
                frame_size=config.frame_size, num_layers=config.num_layers, num_timesteps = config.num_timesteps, bias = config.bias, w_init = config.w_init, hidden_size = config.hidden_size)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay = config.weight_decay)
    model = model.to(device)
    return model, train_loader, test_loader, optimizer, example_data, example_target

def model_pipeline(hyperparameters):
    with wandb.init(project="tc_forecast_prediction_rss", config=hyperparameters):
        config = wandb.config
        random.seed(config.s)
        model, train_loader, test_loader, optimizer, example_data, example_target = make(config)
        run_model(model, train_loader, test_loader, optimizer, config, example_data, example_target)
        return model

if __name__=='__main__':
    
    model = model_pipeline(config)
