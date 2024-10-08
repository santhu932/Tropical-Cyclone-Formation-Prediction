import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import Adam
from rss_lstm.seq2seq2 import Seq2Seq
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
import pandas as pd
import random
import math
from lstm import data_processing
import wandb
import metric
#import visualize
import data

config = dict(epochs = 50,
                clip = 0.0,
                s = 10,
                num_channels = 13,
                num_kernels = 64,
                kernel_size = (3, 3),
                padding = (1, 1),
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
                sampling_step2 = 30,
                attention_hidden_dims = 4,
                future_interval = 1,
                forecasting = False,
                pred_thresold = 0.3,
                iou_threshold = 0.1
                )
              
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, grid_labels):
        self.data = data
        self.labels = labels
        self.grid_labels = grid_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        grid_label = self.grid_labels[idx]
        return sample, label, grid_label

    
    
def train_epoch(model, optimizer, train_loader, epoch, config):
    training_loss = 0.
    accs = []
    batch_count = 0
    examples_count = 0
    model.train()
    for i, (inputs, labels, grid_labels) in enumerate(train_loader, 1):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        inputs, labels, grid_labels = inputs.float().to(device), labels.float().to(device), grid_labels.to(device)
        #inputs, labels, target_output_frames = inputs.float().to(device), labels.float().to(device), target_output_frames.float().to(device)
        #RSS
#        prob_mask = torch.rand(labels.shape[0], config.num_timesteps - 1)
#        prob_mask1 = torch.rand(labels.shape[0], 1)
#        prob_mask, prob_mask1 = prob_mask.to(device), prob_mask1.to(device)
#        if epoch < config.sampling_step1:
#            prob_mask = (prob_mask < config.threshold).float()
#            prob_mask1 = (prob_mask1 < config.threshold).float()
#        elif epoch < config.sampling_step2:
#            prob_mask = (prob_mask < (1 - config.threshold * (config.threshold_decay ** (epoch - config.sampling_step1 + 1)))).float()
#            prob_mask1 = (prob_mask1 < (1 - config.threshold * (config.threshold_decay ** (epoch - config.sampling_step1 + 1)))).float()
#        else:
#            prob_mask = (prob_mask < 1.0).float()
#            prob_mask1 = (prob_mask1 < 1.0).float()
#        prob_mask = prob_mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, config.num_channels, -1, config.frame_size[0], config.frame_size[1])
#        prob_mask1 = prob_mask1.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, config.num_channels, -1, config.frame_size[0], config.frame_size[1])

        outputs, pred_grid_labels = model(inputs.float())
        #outputs, output_frames = model(inputs.float(), prediction = True)
        log_probs = torch.sigmoid(outputs)
        labels = labels.reshape(labels.shape[0], 1)
        loss_func = torch.nn.BCELoss(reduction='none')
        bce_loss = loss_func(log_probs, labels)
        bce_loss = bce_loss/torch.max(bce_loss)
        label_loss = torch.sum(bce_loss) / labels.shape[0]
        #recon_loss = F.mse_loss(output_frames.float(), target_output_frames.float())
        #recon_loss = recon_loss/config.batch_size
        #y_pred = torch.sigmoid(pred_grid_labels)
        #grid_loss = sigmoid_focal_loss(y_pred.float(), grid_labels.float(), alpha=0.25, gamma=2, reduction='sum') / grid_labels.size(0)
        grid_loss = metric.focal_loss(pred_grid_labels.float(), grid_labels.float())
        loss = grid_loss + label_loss
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

def plot_grid_labels(label, save_path=None):
    plt.imshow(label, interpolation='none', vmin=0., vmax=1., cmap='gray')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_roc_auc_cruve(true_labels, log_prob, save_path):
    fpr, tpr, thresholds = roc_curve(true_labels, log_prob)
    auc_score = roc_auc_score(true_labels, log_prob)
    print(auc_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'k--')  # Plotting the diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(save_path)

    
        
def eval_epoch(model, test_loader, epoch, config):
    test_loss = 0.
    all_predicted_labels = []
    all_labels = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    pred_grid_labels = []
    true_grid_labels = []
    with torch.no_grad():
        for i, (inputs, labels, grid_labels) in enumerate(test_loader, 1):
            #example_data, example_target = example_data.float().to(device), example_target.float().to(device)
            inputs, labels, grid_labels = inputs.float().to(device), labels.float().to(device), grid_labels.to(device)
#            if i == 1:
#                inputs[0] = example_data[0]
#                labels[0] = 1
#                target_output_frames[0] = example_target[0]
            outputs, pred_grid = model(inputs.float())
            log_probs = torch.sigmoid(outputs)
            labels = labels.reshape(labels.shape[0], 1)
            loss_func = torch.nn.BCELoss(reduction='none')
            bce_loss = loss_func(log_probs, labels)
            bce_loss = bce_loss/torch.max(bce_loss)
            label_loss = torch.sum(bce_loss) / labels.shape[0]
#            if epoch == 49 and i == 1:
#                for j in range(labels.shape[0]):
#                    if labels[j] == 1:
#                        print(j)
#                        target_array= target_output_frames.clone().detach().cpu().numpy()
#                        predicted_array = output_frames.clone().detach().cpu().numpy()
#                        target_arrayReshaped_tplus1 = target_array[j,0].reshape(target_array[j,0].shape[0], -1)
#                        predicted_arrayReshaped_tplus1 = predicted_array[j,0].reshape(predicted_array[j,0].shape[0], -1)
#                        target_arrayReshaped_tplus2 = target_array[j,1].reshape(target_array[j,1].shape[0], -1)
#                        predicted_arrayReshaped_tplus2 = predicted_array[j,1].reshape(predicted_array[j,1].shape[0], -1)
#                        np.savetxt('predicted_array_tplus1.txt', predicted_arrayReshaped_tplus1)
#                        np.savetxt('target_array_tplus1.txt', target_arrayReshaped_tplus1)
#                        np.savetxt('predicted_array_tplus2.txt', predicted_arrayReshaped_tplus2)
#                        np.savetxt('target_array_tplus2.txt', target_arrayReshaped_tplus2)
#                        break
                #torch.onnx.export(model, inputs, "model.onnx")
                #wandb.save("model.onnx")

            #recon_loss = F.mse_loss(output_frames.float(), target_output_frames.float())
            #recon_loss = recon_loss/config.batch_size
            #y_pred = torch.sigmoid(pred_grid)
            #grid_loss = sigmoid_focal_loss(y_pred.float(), grid_labels.float(), alpha=0.25, gamma=2, reduction='sum') / grid_labels.size(0)
            grid_loss = metric.focal_loss(pred_grid.float(), grid_labels.float())
            loss =  grid_loss + label_loss
            true_positive += torch.sum((log_probs > 0.5) * (labels == 1))
            true_negative += torch.sum((log_probs <= 0.5) * (labels == 0))
            false_positive += torch.sum((log_probs > 0.5) * (labels == 0))
            false_negative += torch.sum((log_probs <= 0.5) * (labels == 1))
            test_loss += loss.cpu().data.numpy()
            all_predicted_labels.append(log_probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            y_pred = torch.sigmoid(pred_grid)
            if epoch == 49:
                for k in range(grid_labels.shape[0]):
                    if torch.any(grid_labels[k,0] > 0.1).item():
                        plot_grid_labels(grid_labels[k,0].detach().cpu().numpy(), save_path=f'visual_lstm/target/grid_labels-{k}.png')
                    if torch.any(y_pred[k,0] > 0.1).item():
                        plot_grid_labels(y_pred[k,0].detach().cpu().numpy(), save_path=f'visual_lstm/predicted/pred_grid_labels-{k}.png')
            pred_grid_labels.append(y_pred.detach().cpu().numpy())
            true_grid_labels.append(grid_labels.detach().cpu().numpy())
    pred_grid_labels = np.concatenate(pred_grid_labels, axis=0).squeeze()
    true_grid_labels = np.concatenate(true_grid_labels, axis=0).squeeze()
    stat1 = metric.bbiou(pred_grid_labels, true_grid_labels, iou_threshold=config.iou_threshold, pred_threshold = config.pred_thresold)
    all_predicted_labels = np.concatenate(all_predicted_labels, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if epoch == 49:
        plot_roc_auc_cruve(all_labels, all_predicted_labels, save_path= 'visual_lstm/plot.png')
    # accuracy
    acc = (true_positive + true_negative)/(true_negative + true_positive + false_positive + false_negative)
    stat = {'TP': true_positive, 'TN': true_negative, 'FP': false_positive, 'FN': false_negative}
    print(stat)
    print("Spatial metrics:\n")
    print(stat1)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return test_loss / len(test_loader.dataset), acc, precision, recall, f1


def run_model(model, train_loader, test_loader, optimizer, config):
    wandb.watch(model, log="all", log_freq=10)
    epochs = config.epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch, config)
        test_loss, test_acc, precision, recall, f1 = eval_epoch(model, test_loader, epoch, config)
        loss_dict = {'train_loss': train_loss, 'test_loss': test_loss}
        wandb.log(loss_dict)
        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f}\tTest Loss: {:.4f}\tTest Accuracy: {:.4f}'.format(
                epoch, train_loss, train_acc, test_loss, test_acc))
        print('Precision: {:.4f} \tRecall: {:.4f}\tF1: {:.4f}'.format(precision, recall, f1))
     

def make(config):
    #train_npz = np.load('ncep_WP_EP_new_2_binary.npz')
    #data, target = train_npz['data'], train_npz['target']
    #data1 = data[:,:3]
    #data1 = np.concatenate((data[:, :3], data[:, 6:7]), axis=1)
    data1 = data.load_data(correct = True)
    mean = np.mean(data1, axis=(0, 2, 3))
    std = np.std(data1, axis=(0, 2, 3))
    transform = transforms.Normalize(mean, std)
    #data, target = torch.from_numpy(data), torch.from_numpy(target)
    data1 = torch.from_numpy(data1)
    data1 = transform(data1)
    train_data, train_target, _, train_grid_labels, test_data, test_target, _, test_grid_labels = data_processing.load_data1(data1, config)
    trainset = CustomDataset(train_data, train_target, train_grid_labels)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size)
    
    testset = CustomDataset(test_data, test_target, test_grid_labels)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size)

    model = Seq2Seq(num_channels=config.num_channels, num_kernels=config.num_kernels,
                kernel_size=config.kernel_size, padding=config.padding, activation=config.activation,
                frame_size=config.frame_size, num_layers=config.num_layers, num_timesteps = config.num_timesteps,
                future_interval = config.future_interval, bias = config.bias, w_init = config.w_init, hidden_size = config.hidden_size, forecasting = config.forecasting)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay = config.weight_decay)
    model = model.to(device)
    return model, train_loader, test_loader, optimizer

def model_pipeline(hyperparameters):
    with wandb.init(project="tc_forecast_prediction_spatial", config=hyperparameters):
        config = wandb.config
        random.seed(config.s)
        model, train_loader, test_loader, optimizer = make(config)
        run_model(model, train_loader, test_loader, optimizer, config)
        return model

if __name__=='__main__':
    
    model = model_pipeline(config)
